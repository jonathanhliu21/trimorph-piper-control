#!/usr/bin/env python3
"""Show live Quest UDP position and orientation without saving samples.

Run from the repository root while the Quest streams JSON packets::

    python3 scripts/quest/visualize_quest.py

Uses the same UDP worker and raw coordinates as test_quest.py. The controller
origin marks position; red, green, and blue segments show its local X, Y, and Z
axes, rotated by the packet's (qx, qy, qz, qw) quaternion. No coordinate-system
conversion is applied. Close the window or press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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


class LatestPose:
    """Worker session/recorder adapter retaining only the most recent sample."""

    def __init__(self) -> None:
        self.start_ns = time.monotonic_ns()
        self._lock = threading.Lock()
        self._row: dict[str, object] | None = None

    def current(self) -> LatestPose:
        return self

    def elapsed_s(self, sample_ns: int) -> float:
        return (sample_ns - self.start_ns) / 1e9

    def append_quest(self, row: dict[str, object]) -> None:
        with self._lock:
            self._row = row.copy()

    def latest(self) -> dict[str, object] | None:
        with self._lock:
            return None if self._row is None else self._row.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    for name in ("startup_timeout_s", "axis_length"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and greater than zero")
    return args


def rotation_matrix(row: dict[str, object]) -> np.ndarray:
    """Return local-to-world rotation for an XYZW quaternion."""
    quaternion = np.array([row[key] for key in ("qx", "qy", "qz", "qw")], dtype=float)
    scale = np.max(np.abs(quaternion))
    if scale == 0:
        raise ValueError("Zero quaternion: orientation unavailable")
    quaternion /= scale
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _raise_worker_error(errors: queue.Queue[WorkerError]) -> None:
    try:
        error = errors.get_nowait()
    except queue.Empty:
        return
    print(error.traceback_text, end="", file=sys.stderr)
    raise RuntimeError(f"Quest worker failed: {error.message}")


def main() -> int:
    args = parse_args()
    config = QuestViewConfig(args.host, args.port, args.startup_timeout_s)
    shared = SharedState()
    poses = LatestPose()
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
    ax.legend(loc="upper right")
    status = fig.text(0.05, 0.03, "Waiting for Quest UDP data...", family="monospace")
    fig.subplots_adjust(bottom=0.18)
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
            row = poses.latest()
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
                status.set_text(
                    f"seq={row['seq']}   sample age={age:.2f}s"
                    f"{'   STREAM STALE' if age > 1.0 else ''}\n"
                    f"p=({position[0]: .3f}, {position[1]: .3f}, {position[2]: .3f})\n"
                    f"q=({row['qx']: .3f}, {row['qy']: .3f}, "
                    f"{row['qz']: .3f}, {row['qw']: .3f}){orientation_status}"
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
