"""3D view of a dataset's demonstrated flange trajectories, with the live executor pose.

All absolute_action_t episodes are drawn in the Piper base frame (dim lines;
green dot = episode start, red dot = end). The executor adds the measured flange
pose (dot + XYZ triad), its recent trail, the chunk being executed, and the
distance to the nearest demonstrated point. Drag to rotate, right-drag or scroll
to zoom; q/Escape stops the executor like Ctrl-C.

Standalone, to inspect a dataset without hardware:

    python inference/executor/trajectory_view.py data_trimorph_hand_mocap_quest_4
"""
from __future__ import annotations

import argparse
from collections import deque
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

CONTROL_ROOT = Path(__file__).resolve().parents[2]
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from data_collection.collection.tracking_monitor import AXIS_COLORS, project  # noqa: E402

WINDOW_NAME = "Trajectories (q/Esc stops)"
DEFAULT_DATASET = CONTROL_ROOT / "data_trimorph_hand_mocap_quest_4"
TRAIL_LENGTH = 300  # Live poses kept, about 20 s at 15 Hz.
GRID_HALF_M = 0.3
GRID_STEP_M = 0.1
TRIAD_M = 0.05

# BGR colors.
DEMO_COLOR = (95, 95, 95)
START_COLOR = (80, 200, 80)
END_COLOR = (70, 70, 210)
LIVE_COLOR = (0, 215, 255)
CHUNK_COLOR = (255, 120, 255)


def load_trajectories(dataset_root, stride=4):
    """Return each episode's flange XYZ (m, Piper base), every stride-th row plus the last."""
    files = sorted(Path(dataset_root).expanduser().glob("absolute_action_t/*.npy"))
    if not files:
        raise FileNotFoundError(f"No absolute_action_t/*.npy under {dataset_root}")
    trajectories = []
    for path in files:
        xyz = np.load(path, allow_pickle=False)[:, :3].astype(float)
        trajectories.append(np.vstack((xyz[::stride], xyz[-1:])))
    return trajectories


class TrajectoryView:
    """OpenCV window drawn from the caller's loop; the demo layer is cached per view angle."""

    def __init__(self, dataset_root, *, window_name=WINDOW_NAME, size=(860, 680), stride=4, cv2=None):
        if cv2 is None:
            import cv2
        self.cv2, self.window_name = cv2, window_name
        self.width, self.height = size
        self.dataset = Path(dataset_root)
        self.trajectories = load_trajectories(dataset_root, stride)
        self.points = np.vstack(self.trajectories)
        self.episode_of_point = np.concatenate([np.full(len(t), i) for i, t in enumerate(self.trajectories)])
        # Fit the demonstrations and the base origin into the window.
        box = np.vstack((self.points.min(0), self.points.max(0), np.zeros(3)))
        self.origin = (box.min(0) + box.max(0)) / 2
        extent = float(np.linalg.norm(np.ptp(box, axis=0)))
        self.yaw, self.pitch = math.radians(-35), math.radians(25)
        self.scale = 0.8 * min(self.width, self.height) / max(extent, 0.1)
        self._drag = None
        self._background = self._background_key = None
        self.trail = deque(maxlen=TRAIL_LENGTH)
        self.pose = None
        self.chunk = None
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, _param):
        cv2 = self.cv2
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self._drag = (event, x, y, self.yaw, self.pitch, self.scale)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            button, x0, y0, yaw, pitch, scale = self._drag
            if button == cv2.EVENT_LBUTTONDOWN:
                self.yaw = yaw + 0.01 * (x - x0)
                self.pitch = float(np.clip(pitch + 0.01 * (y - y0), -1.55, 1.55))
            else:
                self.scale = float(np.clip(scale * math.exp(-0.01 * (y - y0)), 100.0, 20000.0))
        elif event == getattr(cv2, "EVENT_MOUSEWHEEL", None):
            self.scale = float(np.clip(self.scale * (1.15 if flags > 0 else 1 / 1.15), 100.0, 20000.0))

    def _project(self, points):
        center = (self.width // 2, self.height // 2)
        return project(np.asarray(points, dtype=float) - self.origin, self.yaw, self.pitch, self.scale, center)

    def _demo_layer(self):
        key = (self.yaw, self.pitch, self.scale)
        if key == self._background_key:
            return self._background
        cv2 = self.cv2
        image = np.full((self.height, self.width, 3), 28, dtype=np.uint8)
        # Grid on the base's z=0 plane under the demonstrations, plus the base origin axes.
        grid = np.arange(-GRID_HALF_M, GRID_HALF_M + 1e-9, GRID_STEP_M)
        cx, cy = np.round(self.origin[:2] / GRID_STEP_M) * GRID_STEP_M
        for value in grid:
            for line in (((cx + value, cy - GRID_HALF_M, 0), (cx + value, cy + GRID_HALF_M, 0)),
                         ((cx - GRID_HALF_M, cy + value, 0), (cx + GRID_HALF_M, cy + value, 0))):
                a, b = self._project(line)
                cv2.line(image, tuple(a), tuple(b), (52, 52, 52), 1, cv2.LINE_AA)
        base = self._project(np.zeros(3))[0]
        for axis, (color, name) in enumerate(zip(AXIS_COLORS, "XYZ")):
            tip = self._project(np.eye(3)[axis] * 0.1)[0]
            cv2.line(image, tuple(base), tuple(tip), color, 2, cv2.LINE_AA)
            cv2.putText(image, name, tuple(tip + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(image, "base", tuple(base + (6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1,
                    cv2.LINE_AA)
        projected = [self._project(t) for t in self.trajectories]
        cv2.polylines(image, projected, False, DEMO_COLOR, 1, cv2.LINE_AA)
        for points in projected:
            cv2.circle(image, tuple(points[0]), 2, START_COLOR, -1, cv2.LINE_AA)
            cv2.circle(image, tuple(points[-1]), 2, END_COLOR, -1, cv2.LINE_AA)
        self._background, self._background_key = image, key
        return image

    def set_chunk(self, xyz):
        """Positions (m) of the chunk about to be executed, or None to clear it."""
        self.chunk = None if xyz is None else np.asarray(xyz, dtype=float)[:, :3]

    def set_pose(self, pose):
        """Measured Piper flange pose: XYZ m + Euler xyz rad, or None when unavailable."""
        if pose is None:
            return
        pose = np.asarray(pose, dtype=float)
        self.pose = pose
        self.trail.append(pose[:3])

    def render(self):
        cv2 = self.cv2
        image = self._demo_layer().copy()
        if self.chunk is not None and len(self.chunk):
            points = self._project(self.chunk)
            cv2.polylines(image, [points], False, CHUNK_COLOR, 1, cv2.LINE_AA)
            for point in points:
                cv2.circle(image, tuple(point), 3, CHUNK_COLOR, -1, cv2.LINE_AA)
        if len(self.trail) > 1:
            cv2.polylines(image, [self._project(np.array(self.trail))], False, LIVE_COLOR, 2, cv2.LINE_AA)
        lines = [f"{self.dataset.name}: {len(self.trajectories)} episodes"]
        if self.pose is not None:
            position, rotation = self.pose[:3], Rotation.from_euler("xyz", self.pose[3:]).as_matrix()
            tip0 = self._project(position)[0]
            for axis, color in enumerate(AXIS_COLORS):
                tip = self._project(position + TRIAD_M * rotation[:, axis])[0]
                cv2.line(image, tuple(tip0), tuple(tip), color, 2, cv2.LINE_AA)
            cv2.circle(image, tuple(tip0), 6, LIVE_COLOR, -1, cv2.LINE_AA)
            distances = np.linalg.norm(self.points - position, axis=1)
            nearest = int(np.argmin(distances))
            lines.append(f"EEF xyz mm ({position[0] * 1000:.0f}, {position[1] * 1000:.0f}, "
                         f"{position[2] * 1000:.0f})   nearest demo {distances[nearest] * 1000:.0f} mm "
                         f"(episode {self.episode_of_point[nearest]})")
        else:
            lines.append("EEF pose: none yet")
        for row, text in enumerate(lines):
            cv2.putText(image, text, (12, 24 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
                        cv2.LINE_AA)
        legend = (("demo", DEMO_COLOR), ("start", START_COLOR), ("end", END_COLOR), ("live EEF", LIVE_COLOR),
                  ("chunk", CHUNK_COLOR))
        x = 12
        for label, color in legend:
            cv2.circle(image, (x + 5, self.height - 40), 5, color, -1, cv2.LINE_AA)
            cv2.putText(image, label, (x + 15, self.height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200),
                        1, cv2.LINE_AA)
            x += 30 + 9 * len(label)
        cv2.putText(image, f"drag: rotate   right-drag/scroll: zoom   grid {GRID_STEP_M * 100:g} cm",
                    (12, self.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        return image

    def update(self, pose=None):
        self.set_pose(pose)
        self.cv2.imshow(self.window_name, self.render())
        if self.cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            raise KeyboardInterrupt

    def close(self):
        try:
            self.cv2.destroyWindow(self.window_name)
        except Exception:
            pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_root", nargs="?", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--stride", type=int, default=4, help="Draw every n-th row of each episode")
    args = p.parse_args(argv)
    view = TrajectoryView(args.dataset_root, stride=args.stride)
    print(f"{len(view.trajectories)} episodes from {args.dataset_root}; q/Escape closes.")
    try:
        while True:
            view.update()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        view.close()


if __name__ == "__main__":
    main()
