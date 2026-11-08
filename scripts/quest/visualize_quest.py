#!/usr/bin/env python3
"""Show Quest position and orientation from UDP or a recorded CSV.

Run from the repository root while the Quest streams JSON packets::

    python3 scripts/quest/visualize_quest.py
    python3 scripts/quest/visualize_quest.py --from-csv output/calib/coords.csv

OpenCV UI matches the accuracy viewer: drag to rotate, right-drag/scroll to
zoom, R/double-click to center on the controller. Live mode waits for data.
CSV playback accepts calibration (quest_x/.../quest_qw) or raw worker
(px/.../qw) columns with elapsed_s or monotonic_ns timestamps. Click the time
trace to seek, Space to pause/resume, and arrow keys to step samples. J/K or
the on-screen buttons seek previous/next jumps.
The trajectory and XYZ time traces highlight position steps above
--jump-threshold (default 0.05 incoming position units). --speed controls playback.

Uses the same UDP worker and raw coordinates as test_quest.py. The controller
origin marks position; red, green, and blue segments show its local X, Y, and Z
axes, rotated by the packet's (qx, qy, qz, qw) quaternion in Unity coordinates:
X right, Y up, Z forward (left-handed). Only the final screen projection swaps
Unity Y/Z for the collection renderer's Z-up view; raw values stay in Unity axes.
Live mode counts significant position steps between every
received packet, flashes a red alert, and marks the latest jump segment. The
threshold is a distance in incoming units, not a tracking-failure diagnosis.
The controller body is schematic. Close the window or press Q/Esc or Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_collection.collection.config import (
    DEFAULT_QUEST_HOST,
    DEFAULT_QUEST_PORT,
)
from data_collection.collection.quest_worker import run_quest_worker
from data_collection.collection.shared_state import SharedState
from data_collection.collection.utils import WorkerError, worker_wrapper
from data_collection.collection.tracking_monitor import AXIS_COLORS, QUEST_COLOR, TrackingMonitor, project


def project_unity(points, yaw, pitch, scale, center):
    """Display Unity vectors through the Z-up renderer, preserving Y as vertical.

    Apply this after quaternion rotation in Unity axes, to both world and local
    geometry. Swapping the final vectors also handles the handedness difference;
    directly reordering quaternion components would not be equivalent.
    """
    return project(np.asarray(points, dtype=float)[..., [0, 2, 1]], yaw, pitch, scale, center)


@dataclass(frozen=True)
class QuestViewConfig:
    quest_host: str
    quest_port: int
    quest_startup_timeout_s: float | None


@dataclass(frozen=True)
class PositionJump:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    distance: float
    monotonic_ns: int
    interval_s: float


class LatestPose:
    """Count jumps per received packet while retaining only the latest pose/jump."""

    def __init__(self, jump_threshold: float = 0.05) -> None:
        if not math.isfinite(jump_threshold) or jump_threshold <= 0:
            raise ValueError("jump_threshold must be finite and positive")
        self.jump_threshold = jump_threshold
        self.jump_count = 0
        self._last_jump: PositionJump | None = None
        self.start_ns = time.monotonic_ns()
        self._lock = threading.Lock()
        self._row: dict[str, object] | None = None
        self._history = deque(maxlen=3000)

    def current(self) -> LatestPose:
        return self

    def elapsed_s(self, sample_ns: int) -> float:
        return (sample_ns - self.start_ns) / 1e9

    def append_quest(self, row: dict[str, object]) -> None:
        with self._lock:
            if self._row is not None:
                start = tuple(float(self._row[key]) for key in ("px", "py", "pz"))
                end = tuple(float(row[key]) for key in ("px", "py", "pz"))
                distance = math.dist(start, end)
                if distance > self.jump_threshold:
                    self.jump_count += 1
                    self._last_jump = PositionJump(
                        start=start, end=end, distance=distance,
                        monotonic_ns=int(row["monotonic_ns"]),
                        interval_s=(int(row["monotonic_ns"]) - int(self._row["monotonic_ns"])) / 1e9,
                    )
            self._row = row.copy()
            self._history.append(self._row)

    def history_snapshot(self):
        with self._lock:
            return list(self._history), self.jump_count, self._last_jump

    def snapshot(self) -> tuple[dict[str, object] | None, int, PositionJump | None]:
        with self._lock:
            return (None if self._row is None else self._row.copy(), self.jump_count, self._last_jump)

    def latest(self) -> dict[str, object] | None:
        with self._lock:
            return None if self._row is None else self._row.copy()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-csv", type=Path, help="Play a CSV instead of opening the UDP receiver")
    parser.add_argument("--speed", type=float, default=1.0, help="CSV playback speed multiplier (default: 1)")
    parser.add_argument("--jump-threshold", type=float, default=0.05,
                        help="Flag live/CSV position steps above this distance in incoming units (default: 0.05)")
    parser.add_argument("--host", default=DEFAULT_QUEST_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_QUEST_PORT)
    parser.add_argument(
        "--startup-timeout-s", type=float, default=None,
        help="Optional startup deadline; default: wait indefinitely"
    )
    parser.add_argument(
        "--axis-length", type=float, default=0.2,
        help="Length of each controller axis in incoming position units (default: 0.2).",
    )
    args = parser.parse_args(argv)
    if not args.host:
        parser.error("--host must not be empty")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in range 1..65535")
    for name in ("startup_timeout_s", "axis_length", "speed", "jump_threshold"):
        value = getattr(args, name)
        if name == "startup_timeout_s" and value is None:
            continue
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and greater than zero")
    return args


def rotation_matrix(row: dict[str, object]) -> np.ndarray:
    """Return local-to-world rotation for an XYZW quaternion."""
    quaternion = np.array([row[key] for key in ("qx", "qy", "qz", "qw")], dtype=float)
    scale = np.max(np.abs(quaternion))
    if not np.isfinite(quaternion).all():
        raise ValueError("Nonfinite quaternion: orientation unavailable")
    if scale == 0:
        raise ValueError("Zero quaternion: orientation unavailable")
    quaternion /= scale
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def load_csv_poses(path: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Normalize raw-worker and calibration CSVs into the live viewer schema."""
    with path.expanduser().open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        keys = ("px", "py", "pz", "qx", "qy", "qz", "qw")
        if all(f"quest_{key}" in columns for key in ("x", "y", "z", "qx", "qy", "qz", "qw")):
            source = [f"quest_{key}" for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        elif set(keys) <= columns:
            source = list(keys)
        else:
            raise ValueError("CSV needs quest_x/y/z and quest_qx/qy/qz/qw, or px/py/pz and qx/qy/qz/qw")
        clock = "elapsed_s" if "elapsed_s" in columns else "monotonic_ns"
        if clock not in columns:
            raise ValueError("CSV needs elapsed_s or monotonic_ns timestamps")
        rows, stamps = [], []
        for line, raw in enumerate(reader, start=2):
            try:
                row = {key: float(raw[field]) for key, field in zip(keys, source)}
                if not all(math.isfinite(value) for value in row.values()):
                    raise ValueError("Pose values must be finite")
                rotation_matrix(row)  # Validate orientation, including zero quaternions.
                stamp = float(raw[clock]) if clock == "elapsed_s" else int(raw[clock])
                if not math.isfinite(stamp):
                    raise ValueError("Timestamp must be finite")
                row["seq"] = raw.get("seq", raw.get("sample_idx", len(rows)))
                rows.append(row)
                stamps.append(stamp)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}: CSV line {line}: {exc}") from exc
    if len(rows) < 2:
        raise ValueError("CSV needs at least two poses")
    # Subtract integer timestamps before converting to seconds to retain precision.
    times = np.array([stamp - stamps[0] for stamp in stamps], dtype=float)
    if clock == "monotonic_ns":
        times /= 1e9
    if np.any(np.diff(times) <= 0):
        raise ValueError("CSV timestamps must be strictly increasing")
    return times, rows


class Playback:
    """Playback clock and sample/jump navigation, independent of the UI."""

    def __init__(self, times, jumps, speed):
        self.times, self.jumps, self.speed = times, jumps, speed
        self.value = 0.
        self.playing = True
        self.last_wall = time.monotonic()

    @property
    def index(self):
        return max(0, min(len(self.times) - 1, int(np.searchsorted(self.times, self.value, side="right")) - 1))

    def seek(self, value):
        self.value = float(np.clip(value, 0, self.times[-1]))
        self.playing = False
        self.last_wall = time.monotonic()

    def step(self, direction):
        self.seek(self.times[int(np.clip(self.index + direction, 0, len(self.times) - 1))])

    def jump(self, direction):
        candidates = self.jumps[self.jumps > self.index] if direction > 0 else self.jumps[self.jumps < self.index]
        if len(candidates):
            self.seek(self.times[candidates[0] if direction > 0 else candidates[-1]])
        self.playing = False

    def toggle(self):
        if not self.playing and self.value >= self.times[-1]:
            self.value = 0.
        self.playing = not self.playing
        self.last_wall = time.monotonic()

    def tick(self, now):
        elapsed = now - self.last_wall
        self.last_wall = now
        if self.playing:
            self.value = min(float(self.times[-1]), self.value + elapsed * self.speed)
            if self.value >= self.times[-1]:
                self.playing = False


class QuestView(TrackingMonitor):
    """Collection's OpenCV controls with Unity Y-up position and orientation."""

    def __init__(self, shared, args, playback=None):
        self.args, self.playback = args, playback
        self.display_origin = None
        self.current_position = None
        super().__init__(shared, show_quest=True, show_vicon=False,
                         window_name="quest_controller", size=(1200, 900))
        self.scale = 1100.
        self.timeline = None

    def _on_mouse(self, event, x, y, flags, param):
        cv2 = self.cv2
        if self.playback is not None and event == cv2.EVENT_LBUTTONDOWN:
            if 660 <= y <= 810 and 60 <= x <= 1140:
                self.playback.seek((x - 60) / 1080 * self.playback.times[-1])
                return
            if 820 <= y <= 850:
                if 20 <= x <= 170:
                    self.playback.toggle()
                elif 190 <= x <= 370:
                    self.playback.jump(-1)
                elif 390 <= x <= 570:
                    self.playback.jump(1)
                return
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.recenter()
        else:
            super()._on_mouse(event, x, y, flags, param)

    def recenter(self):
        if self.current_position is not None:
            self.display_origin = self.current_position.copy()

    def draw(self, row, history, jump_count, last_jump=None, *, now_ns=None, status=""):
        cv2 = self.cv2
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        image = np.full((self.height, self.width, 3), 28, dtype=np.uint8)
        red, gray = (60, 60, 255), (200, 200, 200)

        def text(value, y, color=gray, size=.55, x=16):
            cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

        recent_jump = last_jump is not None and (now_ns - last_jump.monotonic_ns) / 1e9 < 1.5
        text("QUEST CONTROLLER - Unity axes: X right / Y up / Z forward", 30, QUEST_COLOR, .65)
        text(f"Position jumps: {jump_count} | threshold > {self.args.jump_threshold:g}" +
             ("   POSITION JUMP!" if recent_jump else ""), 59, red if recent_jump else gray)
        if row is not None:
            self.current_position = np.array([row[k] for k in ("px", "py", "pz")], dtype=float)
            if self.display_origin is None:
                self.display_origin = self.current_position.copy()
            text(f"Position XYZ: ({row['px']:+.4f}, {row['py']:+.4f}, {row['pz']:+.4f})", 88)
            text(f"Quaternion XYZW: ({row['qx']:+.4f}, {row['qy']:+.4f}, {row['qz']:+.4f}, {row['qw']:+.4f})", 115)
            if self.playback is None:
                age = (now_ns - int(row['monotonic_ns'])) / 1e9
                status = f"seq={row['seq']} | sample age={age:.2f}s" + (" | NO SIGNAL" if age > .5 else "")
        else:
            text("Waiting for Quest UDP data...", 88)
        text(status, 143)
        if last_jump is not None:
            text(f"Last jump: {last_jump.distance:.4f} units over {last_jump.interval_s:.4f}s", 170, red)
        center = (600, 415)
        origin = self.display_origin if self.display_origin is not None else np.zeros(3)

        def screen(points):
            return project_unity(np.asarray(points) - origin, self.yaw, self.pitch, self.scale, center)

        for value in np.arange(-.5, .501, .1):
            # Unity ground plane is XZ; Y is vertical.
            for line in (((value, 0, -.5), (value, 0, .5)), ((-.5, 0, value), (.5, 0, value))):
                a, b = project_unity(line, self.yaw, self.pitch, self.scale, center)
                cv2.line(image, tuple(a), tuple(b), (55, 55, 55), 1, cv2.LINE_AA)
        # Fixed reference axes; only the camera or explicit recenter moves them.
        for axis, color in enumerate(AXIS_COLORS):
            a, b = project_unity([np.zeros(3), np.eye(3)[axis] * .1], self.yaw, self.pitch, self.scale, center)
            cv2.line(image, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
            cv2.putText(image, "world " + "XYZ"[axis], tuple(b), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
        positions = np.array([[r[k] for k in ("px", "py", "pz")] for r in history])
        if len(positions) > 1:
            points = screen(positions)
            cv2.polylines(image, [points], False, QUEST_COLOR, 1, cv2.LINE_AA)
            for i in np.flatnonzero(np.linalg.norm(np.diff(positions, axis=0), axis=1) > self.args.jump_threshold):
                cv2.line(image, tuple(points[i]), tuple(points[i + 1]), red, 2, cv2.LINE_AA)
        if last_jump is not None:
            a, b = screen([last_jump.start, last_jump.end])
            cv2.line(image, tuple(a), tuple(b), red, 3, cv2.LINE_AA)
            cv2.drawMarker(image, tuple(b), red, cv2.MARKER_TILTED_CROSS, 14, 2)
        if row is not None:
            p = self.current_position
            try:
                rotation = rotation_matrix(row)
                # Schematic controller body: a cuboid, not a calibrated physical model.
                vertices = np.array([[x, y, z] for x in (-.025, .025) for y in (-.018, .018) for z in (-.07, .03)])
                vertices = screen(vertices @ rotation.T + p)
                for i in range(8):
                    for bit in (1, 2, 4):
                        j = i ^ bit
                        if i < j:
                            cv2.line(image, tuple(vertices[i]), tuple(vertices[j]), QUEST_COLOR, 2, cv2.LINE_AA)
                start = screen(p)[0]
                for axis, color in enumerate(AXIS_COLORS):
                    end = screen(p + rotation[:, axis] * self.args.axis_length)[0]
                    cv2.arrowedLine(image, tuple(start), tuple(end), color, 3, cv2.LINE_AA, tipLength=.08)
                    cv2.putText(image, "XYZ"[axis], tuple(end + 5), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
            except ValueError as exc:
                text(str(exc), 197, red)
            cv2.circle(image, tuple(screen(p)[0]), 5, red if recent_jump else QUEST_COLOR, -1, cv2.LINE_AA)
        if self.timeline is not None:
            times, all_positions, jumps = self.timeline
            lower, upper = all_positions.min(), all_positions.max()
            span = max(upper - lower, 1e-6)
            xs = 60 + times / times[-1] * 1080
            cv2.rectangle(image, (60, 660), (1140, 810), (65, 65, 65), 1)
            # Downsample only drawing; playback and jump detection use every sample.
            indices = np.unique(np.linspace(0, len(times) - 1, min(len(times), 2200)).astype(int))
            for axis, color in enumerate(AXIS_COLORS):
                ys = 800 - (all_positions[indices, axis] - lower) / span * 125
                points = np.column_stack((xs[indices], ys)).round().astype(np.int32)
                cv2.polylines(image, [points], False, color, 1, cv2.LINE_AA)
            for index in jumps:
                x = int(xs[index])
                cv2.line(image, (x, 665), (x, 805), red, 1)
            x = int(60 + self.playback.value / times[-1] * 1080)
            cv2.line(image, (x, 660), (x, 810), (255, 255, 255), 2)
            text("XYZ position versus time (click to seek)", 650, size=.45, x=60)
            for x, right, label in ((20, 170, "Pause" if self.playback.playing else "Play"),
                                    (190, 370, "Previous jump"), (390, 570, "Next jump")):
                cv2.rectangle(image, (x, 820), (right, 850), (70, 70, 70), -1)
                text(label, 841, size=.5, x=x + 10)
        text("Drag: rotate | right-drag/scroll: zoom | R/double-click: center on controller | Q/Esc: quit", 875, size=.5)
        text("Local axes: X red, Y green, Z blue | grid: 0.1 units | body is schematic", 620, size=.5)
        return image

    def key(self, key):
        if key in (27, ord('q')):
            return False
        if key == ord('r'):
            self.recenter()
        if self.playback is not None:
            if key == ord(' '):
                self.playback.toggle()
            elif key in (65361, 2424832, 81):
                self.playback.step(-1)
            elif key in (65363, 2555904, 83):
                self.playback.step(1)
            elif key == ord('j'):
                self.playback.jump(-1)
            elif key == ord('k'):
                self.playback.jump(1)
        return True


def visualize_csv(args: argparse.Namespace) -> int:
    times, rows = load_csv_poses(args.from_csv)
    positions = np.array([[row[k] for k in ("px", "py", "pz")] for row in rows])
    steps = np.r_[0., np.linalg.norm(np.diff(positions, axis=0), axis=1)]
    jumps = np.flatnonzero(steps > args.jump_threshold)
    playback = Playback(times, jumps, args.speed)
    view = QuestView(SharedState(), args, playback)
    view.timeline = times, positions, jumps
    print(f"Loaded {len(rows)} poses; {len(jumps)} jumps. Space: play/pause; arrows: step; J/K: previous/next jump.")
    try:
        while True:
            playback.tick(time.monotonic())
            index = playback.index
            previous = jumps[jumps <= index]
            last_jump = None
            if len(previous):
                j = int(previous[-1])
                last_jump = PositionJump(tuple(positions[j-1]), tuple(positions[j]), float(steps[j]),
                                         round(times[j] * 1e9), float(times[j] - times[j-1]))
            # Render the full recorded path, while highlighting the current pose.
            image = view.draw(rows[index], rows, len(jumps), last_jump,
                              now_ns=round(playback.value * 1e9),
                              status=f"CSV {args.from_csv.name} | sample {index}/{len(rows)-1} | "
                                     f"t={playback.value:.3f}s | {args.speed:g}x | step={steps[index]:.4f}")
            view.cv2.imshow(view.window_name, image)
            if not view.key(view.cv2.waitKeyEx(30)):
                break
            if view.cv2.getWindowProperty(view.window_name, view.cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        return 130
    finally:
        view.close()
    return 0


def _raise_worker_error(errors: queue.Queue[WorkerError]) -> None:
    try:
        error = errors.get_nowait()
    except queue.Empty:
        return
    print(error.traceback_text, end="", file=sys.stderr)
    raise RuntimeError(f"Quest worker failed: {error.message}")


def main() -> int:
    args = parse_args()
    if args.from_csv is not None:
        return visualize_csv(args)
    config = QuestViewConfig(args.host, args.port, args.startup_timeout_s)
    shared = SharedState()
    poses = LatestPose(args.jump_threshold)
    errors: queue.Queue[WorkerError] = queue.Queue()
    init_done = threading.Event()
    thread = threading.Thread(
        target=worker_wrapper,
        name="quest-view-worker",
        args=("quest", errors, shared, init_done, run_quest_worker, config, shared, poses),
        daemon=True,
    )

    view = QuestView(shared, args)
    print(f"Waiting for Quest UDP data on {args.host}:{args.port}. Q/Esc quits.")
    thread.start()
    try:
        while not shared.shutdown.is_set():
            _raise_worker_error(errors)
            history, jump_count, last_jump = poses.history_snapshot()
            image = view.draw(history[-1] if history else None, history, jump_count, last_jump)
            view.cv2.imshow(view.window_name, image)
            if not view.key(view.cv2.waitKeyEx(30)):
                break
            if view.cv2.getWindowProperty(view.window_name, view.cv2.WND_PROP_VISIBLE) < 1:
                break
        _raise_worker_error(errors)
    except KeyboardInterrupt:
        return 130
    finally:
        shared.shutdown.set()
        thread.join(timeout=1.)
        view.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
