#!/usr/bin/env python3
"""Show Quest position and orientation from UDP or a recorded CSV.

Run from the repository root while the Quest streams JSON packets::

    python3 scripts/quest/visualize_quest.py
    python3 scripts/quest/visualize_quest.py --from-csv output/calib/coords.csv

CSV playback accepts calibration (quest_x/.../quest_qw) or raw worker
(px/.../qw) columns with elapsed_s or monotonic_ns timestamps. Use the slider
to seek, Space to pause/resume, and arrow keys to step samples while paused.
The trajectory and XYZ time traces highlight position steps above
--jump-threshold (default 0.05 incoming position units). --speed controls playback.

Uses the same UDP worker and raw coordinates as test_quest.py. The controller
origin marks position; red, green, and blue segments show its local X, Y, and Z
axes, rotated by the packet's (qx, qy, qz, qw) quaternion. No coordinate-system
conversion is applied. Live mode counts significant position steps between every
received packet, flashes a red alert, and marks the latest jump segment. The
threshold is a distance in incoming units, not a tracking-failure diagnosis.
Close the window or press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_collection.collection.config import (
    DEFAULT_QUEST_HOST,
    DEFAULT_QUEST_PORT,
    DEFAULT_QUEST_STARTUP_TIMEOUT_S,
)
from data_collection.collection.quest_worker import run_quest_worker
from data_collection.collection.shared_state import SharedState
from data_collection.collection.utils import WorkerError, worker_wrapper


@dataclass(frozen=True)
class QuestViewConfig:
    quest_host: str
    quest_port: int
    quest_startup_timeout_s: float


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

    def snapshot(self) -> tuple[dict[str, object] | None, int, PositionJump | None]:
        with self._lock:
            return (None if self._row is None else self._row.copy(), self.jump_count, self._last_jump)

    def latest(self) -> dict[str, object] | None:
        with self._lock:
            return None if self._row is None else self._row.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-csv", type=Path, help="Play a CSV instead of opening the UDP receiver")
    parser.add_argument("--speed", type=float, default=1.0, help="CSV playback speed multiplier (default: 1)")
    parser.add_argument("--jump-threshold", type=float, default=0.05,
                        help="Flag live/CSV position steps above this distance in incoming units (default: 0.05)")
    parser.add_argument("--host", default=DEFAULT_QUEST_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_QUEST_PORT)
    parser.add_argument(
        "--startup-timeout-s", type=float, default=DEFAULT_QUEST_STARTUP_TIMEOUT_S
    )
    parser.add_argument(
        "--axis-length", type=float, default=0.2,
        help="Length of each controller axis in incoming position units (default: 0.2).",
    )
    args = parser.parse_args()
    if not args.host:
        parser.error("--host must not be empty")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in range 1..65535")
    for name in ("startup_timeout_s", "axis_length", "speed", "jump_threshold"):
        value = getattr(args, name)
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


def visualize_csv(args: argparse.Namespace) -> int:
    times, rows = load_csv_poses(args.from_csv)
    positions = np.array([[row[key] for key in ("px", "py", "pz")] for row in rows])
    steps = np.r_[0., np.linalg.norm(np.diff(positions, axis=0), axis=1)]
    jumps = np.flatnonzero(steps > args.jump_threshold)
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(121, projection="3d")
    trace = fig.add_subplot(122)
    fig.subplots_adjust(bottom=.25, top=.9, wspace=.25)
    fig.suptitle(f"Quest CSV: {args.from_csv.name} — raw coordinates")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    ax.plot(*positions.T, color="gray", alpha=.45, linewidth=1, label="Recorded path")
    for j, index in enumerate(jumps):
        ax.plot(*positions[index-1:index+1].T, color="crimson", linewidth=2,
                label="Position jump" if j == 0 else None)
    point, = ax.plot([], [], [], "ko", label="Current pose")
    axes = [ax.plot([], [], [], color=color, linewidth=3)[0] for color in ("red", "green", "blue")]
    lower, upper = positions.min(axis=0) - args.axis_length, positions.max(axis=0) + args.axis_length
    center, radius = (lower + upper) / 2, np.max(upper - lower) * .55
    for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(value-radius, value+radius)
    ax.legend(loc="upper left")
    for column, (name, color) in enumerate(zip("XYZ", ("red", "green", "blue"))):
        trace.plot(times, positions[:, column], color=color, label=name, linewidth=1)
    if len(jumps):
        # One collection keeps recordings with many flagged steps responsive.
        trace.vlines(times[jumps], positions.min(), positions.max(), color="crimson", alpha=.3,
                     linewidth=.7, label="Position jumps")
    cursor = trace.axvline(0, color="black", linestyle="--")
    trace.set_xlabel("Time since first CSV sample (s)")
    trace.set_ylabel("Quest position (incoming units)")
    trace.legend()
    trace.grid(alpha=.2)
    slider = Slider(fig.add_axes([.13, .14, .7, .035]), "Time (s)", 0., float(times[-1]), valinit=0.)
    toggle = Button(fig.add_axes([.13, .075, .11, .04]), "Pause")
    previous = Button(fig.add_axes([.27, .075, .13, .04]), "Previous jump")
    following = Button(fig.add_axes([.43, .075, .13, .04]), "Next jump")
    status = fig.text(.13, .015, "", family="monospace", fontsize=9)
    state = {"playing": True, "last_wall": time.monotonic(), "index": 0}

    def render(value):
        index = min(int(np.searchsorted(times, value, side="right") - 1), len(rows)-1)
        index = max(0, index)
        state["index"] = index
        row, position = rows[index], positions[index]
        point.set_data_3d(*position[:, None])
        endpoints = position[:, None] + args.axis_length * rotation_matrix(row)
        for column, line in enumerate(axes):
            line.set_data_3d(*np.column_stack((position, endpoints[:, column])))
        cursor.set_xdata([times[index], times[index]])
        status.set_text(f"sample={index}/{len(rows)-1}  t={times[index]:.3f}s  speed={args.speed:g}x  "
                        f"step={steps[index]:.4f}  flagged={len(jumps)} "
                        f"(threshold={args.jump_threshold:g})"
                        f"{'  POSITION JUMP' if steps[index] > args.jump_threshold else ''}"
                        f"\np=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})  "
                        "Space: pause/resume | Left/Right: step | click time trace: seek")
        fig.canvas.draw_idle()

    def set_playing(playing):
        state["playing"] = playing
        state["last_wall"] = time.monotonic()
        toggle.label.set_text("Pause" if playing else "Play")

    def play_pause(_event):
        if not state["playing"] and slider.val >= times[-1]:
            slider.set_val(0.)
        set_playing(not state["playing"])

    def seek_jump(direction):
        set_playing(False)
        candidates = jumps[jumps > state["index"]] if direction > 0 else jumps[jumps < state["index"]]
        if len(candidates):
            slider.set_val(float(times[candidates[0] if direction > 0 else candidates[-1]]))

    def keypress(event):
        if event.key == " ":
            play_pause(event)
        elif event.key in ("left", "right"):
            set_playing(False)
            index = np.clip(state["index"] + (1 if event.key == "right" else -1), 0, len(rows)-1)
            slider.set_val(float(times[index]))

    def click(event):
        if event.inaxes is trace and event.xdata is not None:
            set_playing(False)
            slider.set_val(float(np.clip(event.xdata, 0, times[-1])))
        elif event.inaxes is slider.ax:
            set_playing(False)

    slider.on_changed(render)
    toggle.on_clicked(play_pause)
    previous.on_clicked(lambda _event: seek_jump(-1))
    following.on_clicked(lambda _event: seek_jump(1))
    fig.canvas.mpl_connect("key_press_event", keypress)
    fig.canvas.mpl_connect("button_press_event", click)
    render(0.)
    print(f"Loaded {len(rows)} poses over {times[-1]:.2f}s; "
          f"{len(jumps)} steps exceed {args.jump_threshold:g} position units. "
          "Space pauses; arrow keys step; Previous/Next jump seeks flagged samples.")
    plt.show(block=False)
    try:
        while plt.fignum_exists(fig.number):
            now = time.monotonic()
            elapsed = now - state["last_wall"]
            state["last_wall"] = now
            if state["playing"]:
                slider.set_val(min(float(times[-1]), slider.val + elapsed * args.speed))
                if slider.val >= times[-1]:
                    set_playing(False)
            plt.pause(1 / 30)
    except KeyboardInterrupt:
        return 130
    finally:
        plt.close(fig)
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

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Quest controller pose — raw stream coordinates")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    ax.scatter([0], [0], [0], color="gray", marker="+", label="World origin")
    point, = ax.plot([], [], [], "ko", label="Controller position")
    axes = [
        ax.plot([], [], [], color=color, linewidth=3, label=f"Controller {label}")[0]
        for color, label in zip(("red", "green", "blue"), ("X", "Y", "Z"))
    ]
    jump_line, = ax.plot([], [], [], color="crimson", linestyle="--", marker="x",
                         linewidth=2, markersize=7, label="Latest significant jump")
    ax.legend(loc="upper right")
    alert = fig.text(0.05, 0.95, f"Significant jumps: 0  |  threshold > {args.jump_threshold:g}",
                     fontsize=12, weight="bold", color="black")
    status = fig.text(0.05, 0.03, "Waiting for Quest UDP data...", family="monospace")
    fig.subplots_adjust(bottom=0.20, top=0.88)
    lower = np.full(3, -args.axis_length)
    upper = np.full(3, args.axis_length)
    fig.canvas.mpl_connect("close_event", lambda _event: shared.shutdown.set())

    print(f"Waiting up to {args.startup_timeout_s:g}s for Quest UDP data on "
          f"{args.host}:{args.port}. Close the window or press Ctrl-C to stop.")
    plt.show(block=False)
    thread.start()
    try:
        while plt.fignum_exists(fig.number) and not shared.shutdown.is_set():
            _raise_worker_error(errors)
            row, jump_count, last_jump = poses.snapshot()
            recent_jump = last_jump is not None and (time.monotonic_ns() - last_jump.monotonic_ns) / 1e9 < 1.5
            alert.set_text(f"Significant jumps: {jump_count}  |  threshold > {args.jump_threshold:g}"
                           + ("  POSITION JUMP!" if recent_jump else ""))
            alert.set_color("crimson" if recent_jump else "black")
            point.set_color("crimson" if recent_jump else "black")
            if last_jump is not None:
                segment = np.array([last_jump.start, last_jump.end])
                jump_line.set_data_3d(*segment.T)
                lower = np.minimum(lower, segment.min(axis=0) - args.axis_length)
                upper = np.maximum(upper, segment.max(axis=0) + args.axis_length)
            if row is not None:
                position = np.array([row[key] for key in ("px", "py", "pz")], dtype=float)
                point.set_data_3d(*position[:, None])
                orientation_status = ""
                try:
                    endpoints = position[:, None] + args.axis_length * rotation_matrix(row)
                except ValueError as exc:
                    orientation_status = f"\n{exc}"
                    endpoints = np.repeat(position[:, None], 3, axis=1)
                for index, line in enumerate(axes):
                    line.set_data_3d(*np.column_stack((position, endpoints[:, index])))
                lower = np.minimum(lower, position - args.axis_length)
                upper = np.maximum(upper, position + args.axis_length)
                center = (lower + upper) / 2
                radius = np.max(upper - lower) * 0.55
                for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
                    setter(value - radius, value + radius)
                age = (time.monotonic_ns() - int(row["monotonic_ns"])) / 1e9
                jump_status = (f"\nLast jump: {last_jump.distance:.4f} units over "
                               f"{last_jump.interval_s:.3f}s at t={poses.elapsed_s(last_jump.monotonic_ns):.3f}s"
                               if last_jump is not None else "\nLast jump: none")
                status.set_text(
                    f"seq={row['seq']}   sample age={age:.2f}s"
                    f"{'   STREAM STALE' if age > 1.0 else ''}\n"
                    f"p=({position[0]: .3f}, {position[1]: .3f}, {position[2]: .3f})\n"
                    f"q=({row['qx']: .3f}, {row['qy']: .3f}, "
                    f"{row['qz']: .3f}, {row['qw']: .3f}){orientation_status}{jump_status}"
                )
            fig.canvas.draw_idle()
            plt.pause(1 / 30)
        _raise_worker_error(errors)
    except KeyboardInterrupt:
        return 130
    finally:
        shared.shutdown.set()
        thread.join(timeout=1.0)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
