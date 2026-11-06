#!/usr/bin/env python3
"""Compare configured and live Piper poses with starts in a processed dataset.

Example::

    .venv/bin/python scripts/dataset/visualize_start_positions.py \
        data_trimorph_unscrew_cap_0 --home home_unscrew1 --live --piper-channel can0

Uses the collection monitor's OpenCV styling, projection and axis colors.
Coordinates are calibrated flange poses in Piper base axes. Each episode has
equal weight; default starts are the first processed pose, as in the start audit.
By default this is offline. --live opens a receive-only CAN listener for Piper
joint feedback (0x2A5..0x2A7), then computes the flange pose locally. It never
sends CAN frames, enables/disables actuators, changes settings, or moves hardware.
--output saves an offline PNG without opening a window.
The larger view includes Euler XYZ roll/pitch/yaw histograms in degrees, with
the central 90% band and live per-axis percentiles. --fullscreen or F expands
the window. RPY is unwrapped near the dataset mean for display; orientation
distances use relative rotations rather than Euler-coordinate differences.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import struct
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_collection.collection.config import DEFAULT_EXECUTOR_START, DEFAULT_PIPER_CHANNEL, PIPER_START_JOINTS_DEG
from data_collection.collection.tracking_monitor import AXIS_COLORS, VICON_COLOR, project
from data_collection.collection.utils import import_cv2

HOME_COLOR = (235, 140, 30)  # Same blue as the collection IK monitor's start marker.
SELECTED_COLOR = (0, 165, 255)
TEXT = (230, 230, 230)
LIVE_COLOR = (220, 80, 220)


class PiperFeedback:
    """Only receive/shutdown on the bus; no robot control API is instantiated.

    Piper feedback uses signed big-endian int32 pairs in millidegrees, matching
    pyAgxArm's decode_2A5_joint_12/decode_2A6_joint_34/decode_2A7_joint_56.
    Require fresh timestamps for ALL three pairs, rather than accepting zeros
    or mixing fresh wrist data with stale shoulder data.
    """
    ids = (0x2A5, 0x2A6, 0x2A7)

    def __init__(self, channel, *, bus=None):
        if bus is None:
            import can
            bus = can.Bus(interface="socketcan", channel=channel,
                          can_filters=[{"can_id": i, "can_mask": 0x7ff, "extended": False}
                                       for i in self.ids], receive_own_messages=False)
        self.bus = bus
        self.joints_deg = np.zeros(6)
        self.timestamps = np.full(3, np.nan)

    def poll(self, now=None):
        # Nonblocking and bounded so a busy CAN bus cannot freeze the GUI.
        for _ in range(1000):
            message = self.bus.recv(timeout=0)
            if message is None:
                break
            if (message.arbitration_id not in self.ids or message.is_extended_id
                    or message.is_error_frame or message.is_remote_frame or len(message.data) != 8):
                continue
            pair = self.ids.index(message.arbitration_id)
            self.joints_deg[2 * pair:2 * pair + 2] = np.asarray(struct.unpack('>ii', message.data)) * .001
            self.timestamps[pair] = message.timestamp
        now = time.time() if now is None else now
        ages = now - self.timestamps
        if not np.isfinite(ages).all():
            return None, "LIVE: waiting for all six Piper joints"
        if np.any(ages < -.05) or np.any(ages > .25):
            return None, "LIVE: feedback stale / timestamp invalid (marker hidden)"
        return home_pose(self.joints_deg), "LIVE: receiving Piper joints (read-only)"

    def close(self):
        self.bus.shutdown()


def home_pose(joints_deg):
    """Numerical FK only; deliberately independent of replay/control entry points."""
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

    pose = np.asarray(fk_from_mdh(list(get_mdh('piper')), np.radians(joints_deg).tolist()))
    return np.r_[pose[:3], Rotation.from_euler("xyz", pose[3:]).as_rotvec()]


@dataclass(frozen=True)
class Starts:
    episodes: np.ndarray
    poses: np.ndarray


def load_starts(dataset: Path, sample: str = "first") -> Starts:
    """Read only each chosen row; refuse relative/unknown coordinate frames."""
    paths = sorted((dataset / "absolute_action_t").glob("*.npy"))
    if not paths:
        raise ValueError(f"No absolute_action_t/*.npy episodes found in {dataset}")
    ids, poses = [], []
    for path in paths:
        try:
            episode = int(path.stem)
            with np.load(dataset / "syncs" / f"{path.stem}.npz", allow_pickle=False) as sync:
                if str(sync["pose_frame"]) != "piper_base":
                    raise ValueError("requires pose_frame=piper_base; reprocess relative poses first")
                index = 0
                if sample == "sync":
                    indices = sync["pose_idx"]
                    if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu":
                        raise ValueError("pose_idx must be a nonempty integer vector")
                    index = int(indices[0])
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if data.ndim != 2 or data.shape[1] != 6 or not 0 <= index < len(data):
                raise ValueError("expected nonempty [T,6] poses and a valid start index")
            pose = np.array(data[index], dtype=float)
            if not np.isfinite(pose).all():
                raise ValueError("start pose contains nonfinite values")
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        ids.append(episode)
        poses.append(pose)
    order = np.argsort(ids)
    return Starts(np.asarray(ids)[order], np.asarray(poses)[order])


class StartViewer:
    width, height = 1800, 1100

    def __init__(self, starts: Starts, home: np.ndarray, label: str, dataset: str, sample: str):
        self.cv2 = import_cv2()
        self.starts, self.home, self.label = starts, home, label
        self.dataset, self.sample = dataset, sample
        self.origin = np.median(starts.poses[:, :3], axis=0)
        self.rotations = Rotation.from_rotvec(starts.poses[:, 3:])
        self.mean_rotation = self.rotations.mean()
        self.mean_rpy = self.mean_rotation.as_euler("xyz", degrees=True)
        self.rpy_values = self.display_rpy(self.rotations)
        self.distances = np.linalg.norm(starts.poses[:, :3] - home[:3], axis=1) * 1000
        self.angles = np.degrees((self.rotations * Rotation.from_rotvec(home[3:]).inv()).magnitude())
        self.selected = int(np.argmin(self.distances))
        self.drag = None
        self.live_enabled = False
        self.live_pose = None
        self.live_status = "LIVE: waiting for Piper feedback"
        self.reset()

    def display_rpy(self, rotations):
        """Euler XYZ degrees unwrapped around the dataset SO(3) mean for display.

        Wrapping near +/-180 must not split a compact roll distribution into
        opposite ends of a histogram. Actual angular distances still use SO(3).
        """
        angles = rotations.as_euler("xyz", degrees=True)
        return self.mean_rpy + (angles - self.mean_rpy + 180) % 360 - 180

    def update_live(self, pose, status):
        first = self.live_pose is None
        self.live_pose, self.live_status = pose, status
        if pose is not None and (first or np.linalg.norm(pose[:3] - self.origin) + .025 > self.radius):
            self.reset()

    def reset(self):
        self.yaw, self.pitch = np.radians([-35., 25.])
        points = np.vstack([self.starts.poses[:, :3], self.home[:3]]) - self.origin
        if self.live_pose is not None:
            points = np.vstack([points, self.live_pose[:3] - self.origin])
        self.radius = max(.04, float(np.max(np.linalg.norm(points, axis=1))) + .025)
        self.scale = 215 / self.radius

    def text(self, image, line, xy, color=TEXT, scale=.75):
        self.cv2.putText(image, line, xy, self.cv2.FONT_HERSHEY_SIMPLEX,
                         scale, color, 1, self.cv2.LINE_AA)

    def mouse(self, event, x, y, flags, _param):
        cv2 = self.cv2
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.reset()
        elif event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and 235 <= y < 780 and x < 1000:
            self.drag = (event, x, y, self.yaw, self.pitch, self.scale)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            button, x0, y0, yaw, pitch, scale = self.drag
            if button == cv2.EVENT_LBUTTONDOWN:
                self.yaw = yaw + .01 * (x - x0)
                self.pitch = float(np.clip(pitch + .01 * (y - y0), -1.55, 1.55))
            else:
                self.scale = float(np.clip(scale * np.exp(np.clip(-.01 * (y - y0), -10, 10)), 50, 20000))
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.scale = float(np.clip(self.scale * (1.15 if flags > 0 else 1 / 1.15), 50, 20000))

    def render(self):
        cv2 = self.cv2
        image = np.full((self.height, self.width, 3), 28, np.uint8)
        cv2.rectangle(image, (0, 0), (self.width, 55), HOME_COLOR, -1)
        self.text(image, f"Dataset starts: {self.dataset} | {len(self.starts.episodes)} episodes | {self.sample} sample",
                  (20, 38), scale=1.)
        self.text(image, f"Blue cross: {self.label}   cyan: episode starts   orange ring: selected   RGB: flange XYZ axes",
                  (20, 88))
        offset = (self.home[:3] - self.origin) * 1000
        angle = np.degrees((Rotation.from_rotvec(self.home[3:]) * self.mean_rotation.inv()).magnitude())
        self.text(image, f"Home minus dataset median XYZ: {offset[0]:+.1f}, {offset[1]:+.1f}, {offset[2]:+.1f} mm"
                  f" | rotation from mean: {angle:.1f} deg", (20, 123))
        i = self.selected
        self.text(image, f"Episode {self.starts.episodes[i]:06d}: {self.distances[i]:.1f} mm / {self.angles[i]:.1f} deg from home"
                  f" | nearest XYZ: {self.starts.episodes[np.argmin(self.distances)]:06d}", (20, 158), SELECTED_COLOR)
        if self.live_enabled:
            if self.live_pose is None:
                self.text(image, self.live_status, (20, 193), LIVE_COLOR)
            else:
                delta = (self.live_pose[:3] - self.origin) * 1000
                distances = np.linalg.norm(self.starts.poses[:, :3] - self.live_pose[:3], axis=1) * 1000
                nearest = int(np.argmin(distances))
                live_rotation = Rotation.from_rotvec(self.live_pose[3:])
                angle = np.degrees((live_rotation * self.mean_rotation.inv()).magnitude())
                nearest_angle = np.degrees((live_rotation * self.rotations[nearest].inv()).magnitude())
                self.text(image, f"LIVE (magenta): median offset {delta[0]:+.1f}, {delta[1]:+.1f}, {delta[2]:+.1f} mm"
                          f" | rotation from mean {angle:.1f} deg", (20, 193), LIVE_COLOR)
                self.text(image, f"Nearest live XYZ: episode {self.starts.episodes[nearest]:06d}, {distances[nearest]:.1f} mm"
                          f" / {nearest_angle:.1f} deg | passive CAN receive only", (20, 228), LIVE_COLOR)
        self.text(image, "3D | Piper base axes | median XYZ origin | 10 mm grid", (20, 265))
        # Draw into a cropped panel so zoomed geometry cannot overwrite the header.
        panel = image[280:775, :1000]
        center = (panel.shape[1] // 2, panel.shape[0] // 2)
        def view(points):
            return project(np.asarray(points) - self.origin, self.yaw, self.pitch, self.scale, center)
        for value in np.arange(-self.radius, self.radius + .001, .01):
            for a, b in (((value, -self.radius, 0), (value, self.radius, 0)),
                         ((-self.radius, value, 0), (self.radius, value, 0))):
                p = view(self.origin + np.array([a, b]))
                cv2.line(panel, tuple(p[0]), tuple(p[1]), (50, 50, 50), 1, cv2.LINE_AA)
        for point in view(self.starts.poses[:, :3]):
            cv2.circle(panel, tuple(point), 3, VICON_COLOR, -1, cv2.LINE_AA)
        markers = [(self.starts.poses[i], SELECTED_COLOR, False), (self.home, HOME_COLOR, True)]
        if self.live_pose is not None:
            markers.append((self.live_pose, LIVE_COLOR, False))
        for pose, color, cross in markers:
            origin = view(pose[:3])[0]
            rotation = Rotation.from_rotvec(pose[3:]).as_matrix()
            for axis, axis_color in enumerate(AXIS_COLORS):
                tip = view(pose[:3] + .02 * rotation[:, axis])[0]
                cv2.line(panel, tuple(origin), tuple(tip), axis_color, 2, cv2.LINE_AA)
                self.text(panel, "XYZ"[axis], tuple(tip + 4), axis_color)
            if cross:
                cv2.drawMarker(panel, tuple(origin), color, cv2.MARKER_CROSS, 20, 3)
            else:
                cv2.circle(panel, tuple(origin), 9, color, 2, cv2.LINE_AA)
        for column, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            self.draw_projection(image[790:1030, column * 600:(column + 1) * 600], a, b)
        for axis in range(3):
            self.draw_orientation(image[240 + axis * 180:415 + axis * 180, 1000:1800], axis)
        self.text(image, "A/D or [/] : previous/next episode   N: nearest XYZ   R/double-click: reset   F: fullscreen   Q/Esc: quit", (20, 1060), scale=.7)
        self.text(image, "Drag: rotate   scroll: zoom | RPY: Euler xyz degrees, unwrapped near mean; bands are descriptive, not success boundaries.", (20, 1090), scale=.65)
        return image

    def draw_orientation(self, panel, axis):
        cv2 = self.cv2
        panel[:] = (36, 36, 36)
        values = self.rpy_values[:, axis]
        p05, median, p95 = np.percentile(values, [5, 50, 95])
        home = self.display_rpy(Rotation.from_rotvec(self.home[3:]))[axis]
        selected = values[self.selected]
        live = None if self.live_pose is None else self.display_rpy(Rotation.from_rotvec(self.live_pose[3:]))[axis]
        self.text(panel, f"{('ROLL', 'PITCH', 'YAW')[axis]} (deg) | 5-95%: {p05:.1f} .. {p95:.1f}", (16, 25), scale=.75)
        target = live if live is not None else home
        percentile = 100 * np.mean(values <= target)
        name = "Live" if live is not None else "Home"
        if self.live_enabled and live is None:
            self.text(panel, "Live: no fresh feedback", (16, 56), LIVE_COLOR, scale=.85)
        else:
            tail = " | outside 5-95%" if target < p05 or target > p95 else ""
            self.text(panel, f"{name}: {target:+.1f} deg | percentile {percentile:.1f}{tail}",
                      (16, 56), LIVE_COLOR if live is not None else HOME_COLOR, scale=.85)
        self.text(panel, f"Home {home:+.1f}    Selected {selected:+.1f}    Median {median:+.1f}", (16, 83), scale=.65)
        low = min(values.min(), home, target) - 2
        high = max(values.max(), home, target) + 2
        def x(value):
            return int(24 + (value - low) / (high - low) * (panel.shape[1] - 48))
        # Shaded central band, cyan histogram, colored home/episode/live lines.
        cv2.rectangle(panel, (x(p05), 93), (x(p95), 146), (65, 55, 45), -1)
        counts, edges = np.histogram(values, bins=24, range=(low, high))
        for count, left, right in zip(counts, edges[:-1], edges[1:]):
            if count:
                height = int(46 * count / counts.max())
                cv2.rectangle(panel, (x(left), 146 - height), (max(x(left), x(right) - 1), 146), VICON_COLOR, -1)
        for value, color in ((home, HOME_COLOR), (selected, SELECTED_COLOR), (live, LIVE_COLOR)):
            if value is not None:
                cv2.line(panel, (x(value), 90), (x(value), 148), color, 3, cv2.LINE_AA)
        self.text(panel, f"{low:.1f}", (24, 168), scale=.55)
        self.text(panel, f"{high:.1f}", (panel.shape[1] - 90, 168), scale=.55)

    def draw_projection(self, panel, a, b):
        cv2 = self.cv2
        points = np.vstack([self.starts.poses[:, :3], self.home[:3]])[:, [a, b]] * 1000
        bounds = points
        if self.live_pose is not None:
            bounds = np.vstack([bounds, self.live_pose[[a, b]] * 1000])
        low, high = bounds.min(axis=0) - 8, bounds.max(axis=0) + 8
        scale = min((panel.shape[1] - 100) / (high[0] - low[0]), 150 / (high[1] - low[1]))
        middle = (low + high) / 2
        def xy(point):
            return tuple(np.rint((np.asarray(point) - middle) * [scale, -scale] + [panel.shape[1] // 2, 125]).astype(int))
        for axis in range(2):
            for value in np.arange(np.ceil(low[axis] / 10) * 10, high[axis], 10):
                start, end = low.copy(), high.copy()
                start[axis] = end[axis] = value
                cv2.line(panel, xy(start), xy(end), (50, 50, 50), 1)
        for point in points[:-1]:
            cv2.circle(panel, xy(point), 3, VICON_COLOR, -1, cv2.LINE_AA)
        cv2.circle(panel, xy(points[self.selected]), 8, SELECTED_COLOR, 2)
        cv2.drawMarker(panel, xy(points[-1]), HOME_COLOR, cv2.MARKER_CROSS, 18, 2)
        if self.live_pose is not None:
            cv2.circle(panel, xy(self.live_pose[[a, b]] * 1000), 9, LIVE_COLOR, 3, cv2.LINE_AA)
        self.text(panel, f"{'XYZ'[a]} / {'XYZ'[b]} (absolute mm)", (20, 28))
        self.text(panel, f"{'XYZ'[a]}: {low[0]:.0f} .. {high[0]:.0f}   {'XYZ'[b]}: {low[1]:.0f} .. {high[1]:.0f}", (20, 233), scale=.65)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--home", choices=sorted(PIPER_START_JOINTS_DEG), default=None,
                           help=f"Configured start preset (default: {DEFAULT_EXECUTOR_START})")
    reference.add_argument("--start-joints-deg", type=float, nargs=6, metavar="DEG",
                           help="Custom J1..J6 reference, converted with offline Piper FK")
    parser.add_argument("--sample", choices=("first", "sync"), default="first",
                        help="First processed pose or pose at first synchronized camera frame")
    parser.add_argument("--output", type=Path, help="Save an image and exit without opening a window")
    parser.add_argument("--live", action="store_true", help="Passively receive current Piper joints over SocketCAN")
    parser.add_argument("--piper-channel", default=DEFAULT_PIPER_CHANNEL, help="Existing CAN interface (default: can0)")
    parser.add_argument("--fullscreen", action="store_true", help="Use the full display; F toggles fullscreen")
    args = parser.parse_args()
    if args.live and args.output:
        parser.error("--output is offline only; use --live for the interactive feedback viewer")
    try:
        starts = load_starts(args.dataset.expanduser(), args.sample)
        name = args.home or DEFAULT_EXECUTOR_START
        joints = np.asarray(args.start_joints_deg if args.start_joints_deg is not None else PIPER_START_JOINTS_DEG[name])
        if not np.isfinite(joints).all():
            raise ValueError("start joints must be finite")
        home = home_pose(joints)
        viewer = StartViewer(starts, home, "custom joints" if args.start_joints_deg is not None else name,
                             args.dataset.name, args.sample)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    cv2 = viewer.cv2
    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), viewer.render()):
            raise OSError(f"Could not write {output}")
        print(f"Saved {output} ({len(starts.episodes)} episode starts)")
        return
    window = "trimorph_dataset_starts"
    feedback = None
    try:
        if args.live:
            feedback = PiperFeedback(args.piper_channel)
            viewer.live_enabled = True
            print(f"Listening on {args.piper_channel}: CAN receive only; no commands or settings sent.")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, viewer.width, viewer.height)
        fullscreen = args.fullscreen
        if fullscreen:
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(window, viewer.mouse)
        while True:
            if feedback is not None:
                viewer.update_live(*feedback.poll())
            cv2.imshow(window, viewer.render())
            key = cv2.waitKey(30) & 0xff
            if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('a'), ord('['), ord('d'), ord(']')):
                viewer.selected = (viewer.selected + (-1 if key in (ord('a'), ord('[')) else 1)) % len(starts.episodes)
            elif key == ord('n'):
                viewer.selected = int(np.argmin(viewer.distances))
            elif key == ord('r'):
                viewer.reset()
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    finally:
        if feedback is not None:
            feedback.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
