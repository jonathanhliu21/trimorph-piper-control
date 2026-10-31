#!/usr/bin/env python3
"""Show detected ROS 2 Vicon objects and markers in an interactive 3D view.

Source ROS 2 and the Vicon workspace, then run:
    python scripts/calib/mocap_detection.py
    python scripts/calib/mocap_detection.py --mocap-topic /your/pose/topic

Object positions use raw receiver coordinates. Red/green/blue axes show local
X/Y/Z from the XYZW quaternion; markers have position only. Drag to rotate the
view. Close the window or press Ctrl-C to stop.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection.collection.mocap_worker import MocapSubscriber


POSE_TYPE = "vicon_receiver/msg/Position"


class DetectedObjects:
    """Keep only the latest observation for each named segment or marker."""

    def __init__(self):
        self.lock = threading.Lock()
        self.objects = {}

    def update(self, row):
        key = tuple(row[field] for field in (
            "topic", "message_type", "subject_name", "segment_name", "marker_name",
        ))
        position = np.array([row[field] for field in ("x_trans", "y_trans", "z_trans")], dtype=float)
        with self.lock:
            if row["occluded"] or not np.isfinite(position).all():
                self.objects.pop(key, None)
            else:
                self.objects[key] = row.copy()

    def snapshot(self, stale_after_s):
        cutoff_ns = time.monotonic_ns() - round(stale_after_s * 1e9)
        with self.lock:
            self.objects = {
                key: row for key, row in self.objects.items()
                if row["monotonic_ns"] >= cutoff_ns
            }
            return list(self.objects.values())


def orientation_axes(row):
    """Return local XYZ axes in receiver coordinates, or None if unavailable."""
    if row["message_type"] != POSE_TYPE:
        return None
    quaternion = np.array([row[field] for field in ("x_rot", "y_rot", "z_rot", "w")], dtype=float)
    if not np.isfinite(quaternion).all():
        return None
    scale = np.max(np.abs(quaternion))
    if scale == 0:
        return None
    quaternion /= scale
    return Rotation.from_quat(quaternion).as_matrix()


class MocapView:
    def __init__(self, figure, *, axis_length=None):
        self.figure = figure
        self.axes = figure.add_subplot(111, projection="3d")
        self.axes.set_xlabel("X (receiver units)")
        self.axes.set_ylabel("Y (receiver units)")
        self.axes.set_zlabel("Z (receiver units)")
        self.axes.set_box_aspect((1, 1, 1))
        self.axis_length = axis_length
        self.artists = []
        self.status = figure.text(0.02, 0.02, "Waiting for mocap topics / samples…", fontsize=9)
        figure.suptitle("Mocap detections — local axes: X red, Y green, Z blue")

    def draw(self, rows):
        for artist in self.artists:
            artist.remove()
        self.artists.clear()
        if not rows:
            self.status.set_text("Waiting for visible mocap objects / markers…")
            return

        positions = np.array([[row[field] for field in ("x_trans", "y_trans", "z_trans")] for row in rows], dtype=float)
        center = (positions.min(axis=0) + positions.max(axis=0)) / 2
        span = max(float(np.ptp(positions, axis=0).max()), float(np.abs(center).max()) * 0.1, 1.0)
        length = self.axis_length if self.axis_length is not None else span * 0.15
        extent = span * 0.6 + length
        for setter, value in zip((self.axes.set_xlim, self.axes.set_ylim, self.axes.set_zlim), center):
            setter(value - extent, value + extent)

        poses = 0
        invalid_orientations = 0
        for row, position in zip(rows, positions):
            is_pose = row["message_type"] == POSE_TYPE
            poses += int(is_pose)
            self.artists.append(self.axes.scatter(*position, color="black" if is_pose else "orange", s=35 if is_pose else 15))
            name = "/".join(str(row[field]) for field in ("subject_name", "segment_name", "marker_name") if row[field])
            label = f"{name or row['topic']}\n({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
            rotation = orientation_axes(row)
            if rotation is None and is_pose:
                invalid_orientations += 1
                label += "\norientation unavailable"
            self.artists.append(self.axes.text(*position, label, fontsize=8))
            if rotation is not None:
                for axis, color in enumerate(("red", "green", "blue")):
                    self.artists.append(self.axes.quiver(
                        *position, *rotation[:, axis], length=length, color=color,
                        arrow_length_ratio=0.2,
                    ))
        frames = ", ".join(sorted({str(row["frame_id"]) or "unspecified" for row in rows}))
        units = ", ".join(sorted({str(row["translation_type"]) for row in rows if row["translation_type"]})) or "receiver units"
        self.status.set_text(
            f"{poses} segment poses; {len(rows) - poses} markers | frame_id: {frames} | translation_type: {units}\n"
            f"Occluded/stale detections hidden. {invalid_orientations} unavailable orientations. Raw coordinates; no frame transforms."
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mocap-topic", "--topic", dest="topics", action="append", default=[],
                        help="Select a Vicon topic (repeatable); default: discover all supported topics.")
    parser.add_argument("--axis-length", type=float,
                        help="Length of orientation arrows in receiver units (default: automatic).")
    parser.add_argument("--stale-after-s", type=float, default=2.,
                        help="Hide detections after this many seconds without updates (default: 2).")
    parser.add_argument("--refresh-hz", type=float, default=20., help="Display refresh rate (default: 20).")
    args = parser.parse_args()
    for name in ("axis_length", "stale_after_s", "refresh_hz"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def main():
    args = parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    objects = DetectedObjects()
    subscriber = MocapSubscriber(objects.update, args.topics)
    stop = threading.Event()
    errors = []
    thread = None
    figure = None

    def receive():
        try:
            while not stop.is_set():
                subscriber.poll()
        except Exception as exc:
            errors.append(exc)
            stop.set()

    try:
        thread = threading.Thread(target=receive, name="mocap-view", daemon=True)
        thread.start()
        figure = plt.figure(figsize=(11, 8))
        view = MocapView(figure, axis_length=args.axis_length)
        figure.subplots_adjust(bottom=0.15, top=0.9)
        figure.canvas.mpl_connect("close_event", lambda event: stop.set())

        def refresh(_frame):
            if errors:
                plt.close(figure)
                return
            view.draw(objects.snapshot(args.stale_after_s))

        # Keep the animation alive for the lifetime of the window.
        animation = FuncAnimation(figure, refresh, interval=1000 / args.refresh_hz,
                                  cache_frame_data=False)
        print("Waiting for ROS 2 Vicon detections. Close the plot or press Ctrl-C to stop.")
        plt.show()
        if errors:
            raise RuntimeError(f"Mocap visualization failed: {errors[0]}") from errors[0]
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        stop.set()
        if thread is not None:
            thread.join()
        subscriber.close()
        if figure is not None:
            plt.close(figure)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
