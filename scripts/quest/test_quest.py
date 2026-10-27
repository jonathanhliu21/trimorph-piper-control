#!/usr/bin/env python3
"""Record and verify Quest UDP poses without connecting to the robot hand.

Run from the repository root while the Quest is streaming JSON packets:

    python3 scripts/quest/test_quest.py

By default, the script listens on UDP port 5005 for five seconds, records ten
seconds of poses, writes ``output/quest_test/quest.csv``, then reads the file
back and verifies that its coordinates match the worker output.
"""

from __future__ import annotations

import argparse
import csv
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_collection.collection.config import (
    DEFAULT_QUEST_HOST,
    DEFAULT_QUEST_PORT,
    DEFAULT_QUEST_STARTUP_TIMEOUT_S,
    QUEST_PACKET_FIELDS,
)
from data_collection.collection.quest_worker import run_quest_worker
from data_collection.collection.shared_state import SharedState
from data_collection.collection.utils import WorkerError, worker_wrapper, write_csv


DEFAULT_OUTPUT = Path("output/quest_test/quest.csv")
DEFAULT_DURATION_S = 10.0


@dataclass(frozen=True)
class QuestTestConfig:
    quest_host: str
    quest_port: int
    quest_startup_timeout_s: float


class QuestTestSession:
    """Minimal recording session implementing the interface used by the worker."""

    def __init__(self) -> None:
        self.start_ns = time.monotonic_ns()
        self._lock = threading.Lock()
        self._rows: list[dict[str, object]] = []

    def elapsed_s(self, sample_ns: int) -> float:
        return (sample_ns - self.start_ns) / 1e9

    def append_quest(self, row: dict[str, object]) -> None:
        with self._lock:
            self._rows.append(row)

    def rows(self) -> list[dict[str, object]]:
        with self._lock:
            return list(self._rows)


class QuestTestRecorder:
    def __init__(self, session: QuestTestSession) -> None:
        self._session = session

    def current(self) -> QuestTestSession:
        return self._session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record Quest UDP poses directly to a CSV without starting the hand, "
            "camera, Piper, or episode collector."
        )
    )
    parser.add_argument("--host", default=DEFAULT_QUEST_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_QUEST_PORT)
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=DEFAULT_QUEST_STARTUP_TIMEOUT_S,
    )
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.host:
        parser.error("--host must not be empty")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in range 1..65535")
    if args.startup_timeout_s <= 0:
        parser.error("--startup-timeout-s must be greater than zero")
    if args.duration_s <= 0:
        parser.error("--duration-s must be greater than zero")
    return args


def _raise_worker_error(errors: queue.Queue[WorkerError]) -> None:
    try:
        error = errors.get_nowait()
    except queue.Empty:
        return
    print(error.traceback_text, end="", file=sys.stderr)
    raise RuntimeError(f"Quest worker failed: {error.message}")


def _verify_csv(path: Path, expected: list[dict[str, object]]) -> None:
    with path.open(newline="", encoding="utf-8") as csv_file:
        actual = list(csv.DictReader(csv_file))

    if len(actual) != len(expected):
        raise RuntimeError(
            f"CSV verification failed: expected {len(expected)} rows, "
            f"read back {len(actual)}"
        )
    required_fields = {
        "sample_idx",
        "monotonic_ns",
        "wall_time_s",
        "elapsed_s",
        *QUEST_PACKET_FIELDS,
    }
    if actual and not required_fields.issubset(actual[0]):
        missing = sorted(required_fields - set(actual[0]))
        raise RuntimeError(
            "CSV verification failed: missing columns " + ", ".join(missing)
        )

    for index, (written, source) in enumerate(zip(actual, expected)):
        for field in QUEST_PACKET_FIELDS:
            if written[field] != str(source[field]):
                raise RuntimeError(
                    f"CSV verification failed at row {index}, field {field}: "
                    f"expected {source[field]!r}, read {written[field]!r}"
                )


def _format_pose(label: str, row: dict[str, object]) -> str:
    return (
        f"{label} seq={row['seq']}: "
        f"p=({row['px']}, {row['py']}, {row['pz']}) "
        f"q=({row['qx']}, {row['qy']}, {row['qz']}, {row['qw']})"
    )


def main() -> int:
    args = parse_args()
    config = QuestTestConfig(
        quest_host=args.host,
        quest_port=args.port,
        quest_startup_timeout_s=args.startup_timeout_s,
    )
    shared = SharedState()
    session = QuestTestSession()
    recorder = QuestTestRecorder(session)
    errors: queue.Queue[WorkerError] = queue.Queue()
    init_done = threading.Event()
    thread = threading.Thread(
        target=worker_wrapper,
        name="quest-test-worker",
        args=(
            "quest",
            errors,
            shared,
            init_done,
            run_quest_worker,
            config,
            shared,
            recorder,
        ),
        daemon=True,
    )

    print(
        f"Waiting up to {args.startup_timeout_s:g}s for Quest UDP data on "
        f"{args.host}:{args.port}..."
    )
    thread.start()
    interrupted = False
    try:
        if not init_done.wait(args.startup_timeout_s + 1.0):
            raise TimeoutError("Quest worker did not finish initialization")
        _raise_worker_error(errors)

        print(
            f"Quest stream detected. Recording for {args.duration_s:g}s; "
            "press Ctrl-C to stop early."
        )
        deadline = time.monotonic() + args.duration_s
        while not shared.shutdown.is_set():
            _raise_worker_error(errors)
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            shared.shutdown.wait(min(remaining_s, 0.1))
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopping early and saving received samples...")
    finally:
        shared.shutdown.set()
        thread.join(timeout=1.0)

    _raise_worker_error(errors)
    rows = session.rows()
    if not rows:
        raise RuntimeError("Quest worker received no recordable pose samples")

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    _verify_csv(output, rows)

    print(f"Saved and verified {len(rows)} Quest poses in {output.resolve()}")
    if len(rows) >= 2:
        average_interval_ms = (
            (int(rows[-1]["monotonic_ns"]) - int(rows[0]["monotonic_ns"]))
            / (len(rows) - 1)
            / 1e6
        )
        print(
            "Average time between Quest frames (host receive timestamps): "
            f"{average_interval_ms:.3f} ms"
        )
        frame_intervals_ms = [
            (int(current["monotonic_ns"]) - int(previous["monotonic_ns"])) / 1e6
            for previous, current in zip(rows, rows[1:])
        ]
        print(
            "Quest frame time min/max (host receive timestamps): "
            f"min={min(frame_intervals_ms):.3f} ms, "
            f"max={max(frame_intervals_ms):.3f} ms"
        )
        percentiles = (50, 90, 95, 99)
        values = np.percentile(frame_intervals_ms, percentiles)
        print(
            "Quest frame time percentiles (host receive timestamps): "
            + ", ".join(
                f"p{percentile}={value:.3f} ms"
                for percentile, value in zip(percentiles, values)
            )
        )
    else:
        print("Average time between Quest frames: unavailable (need at least 2 samples)")
        print("Quest frame time min/max: unavailable (need at least 2 samples)")
        print("Quest frame time percentiles: unavailable (need at least 2 samples)")
    print(_format_pose("First", rows[0]))
    print(_format_pose("Last ", rows[-1]))
    return 130 if interrupted else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
