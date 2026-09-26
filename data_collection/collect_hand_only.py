import argparse
import queue
import sys
import threading
import time

from dataclasses import dataclass, field
from pathlib import Path

from collection.config import TRIMORPH_MOTOR_IDS
from collection.episode import (
    EpisodeBuffer,
    RecordingManager,
    episode_paths,
    next_episode_index,
    raw_root,
)
from collection.proprioception_worker import run_proprioception_worker
from collection.shared_state import SharedState
from collection.utils import WorkerError, worker_wrapper

WORKER_INIT_TIMEOUT_S = 60.0
DEFAULT_DEMONSTRATION_DURATION_S = 30.0
DEFAULT_WARMUP_TIME_S = 5.0


@dataclass(frozen=True)
class HandOnlyCollectionConfig:
    dataset_root: Path
    hand_config: Path
    hand_port: str
    hand_measure_rate_hz: float

    warmup_time: float
    episode_time: float
    active_motors: list[int] = field(
        default_factory=lambda: list(TRIMORPH_MOTOR_IDS)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record one fixed-duration TRIMORPH proprioception episode."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data_trimorph"))
    parser.add_argument("--hand-config", type=Path, default=Path("config/hand.yaml"))
    parser.add_argument(
        "--hand-port", default="/dev/serial/by-id/REPLACE_WITH_TRIMORPH_HAND"
    )
    parser.add_argument(
        "--measure-hz",
        "--hand-measure-rate-hz",
        dest="hand_measure_rate_hz",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--episode-time",
        type=float,
        default=DEFAULT_DEMONSTRATION_DURATION_S,
    )
    parser.add_argument(
        "--warmup-time",
        type=float,
        default=DEFAULT_WARMUP_TIME_S,
    )
    parser.add_argument(
        "--active-motors",
        nargs="+",
        type=int,
        default=list(TRIMORPH_MOTOR_IDS),
    )
    return parser.parse_args()


def initialize_worker(
    config: HandOnlyCollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    errors: queue.Queue[WorkerError],
) -> threading.Thread:
    try:
        init_done = threading.Event()
        thread = threading.Thread(
            target=worker_wrapper,
            name="proprioception-worker",
            args=(
                "proprioception",
                errors,
                shared,
                init_done,
                run_proprioception_worker,
                config,
                shared,
                recorder,
            ),
            daemon=True,
        )
        print("Starting background worker...")

        thread.start()
        if not init_done.wait(timeout=WORKER_INIT_TIMEOUT_S):
            raise TimeoutError(
                f"Proprioception worker did not initialize within {WORKER_INIT_TIMEOUT_S}s"
            )
        if not errors.empty():
            error = errors.get_nowait()
            raise RuntimeError(f"{error.name} worker failed: {error.message}")
    except Exception:
        shared.shutdown.set()
        thread.join(timeout=3.0)
        raise

    return thread


def start_episode(
    config: HandOnlyCollectionConfig, recorder: RecordingManager
) -> EpisodeBuffer:
    index = next_episode_index(raw_root(config.dataset_root))
    paths = episode_paths(config.dataset_root, index)
    paths.episode_dir.mkdir(parents=True, exist_ok=False)
    session = EpisodeBuffer(paths, config)
    recorder.start(session)
    print(f"Recording episode {index:06d} for {config.episode_time:.1f}s...")
    return session


def stop_episode(recorder: RecordingManager, reason: str) -> EpisodeBuffer | None:
    session = recorder.stop(reason)
    if session is None:
        return None
    session.flush()
    print(
        f"Saved episode {session.paths.episode_index:06d}: "
        f"{len(session.proprioception_rows)} proprioception samples"
    )
    return session


def run_hand_only(config: HandOnlyCollectionConfig) -> int:
    shared = SharedState()
    recorder = RecordingManager()
    errors: queue.Queue[WorkerError] = queue.Queue()

    print(f"Warming up for {config.warmup_time}s")
    time.sleep(config.warmup_time)
    print("Warmup done, starting data collection...")

    thread = initialize_worker(config, shared, recorder, errors)
    reason = "duration_complete"
    try:
        start_episode(config, recorder)
        shared.shutdown.wait(config.episode_time)
        if not errors.empty():
            error = errors.get_nowait()
            session = recorder.current()
            if session is not None:
                session.add_error(error)
            reason = "proprioception_worker_error"
            raise RuntimeError(f"{error.name} worker failed: {error.message}")
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
        print("\nCollection interrupted; saving recorded samples.")
    finally:
        shared.shutdown.set()
        thread.join(timeout=3.0)
        stop_episode(recorder, reason)
    return 0


def main() -> int:
    args = parse_args()
    config = HandOnlyCollectionConfig(
        dataset_root=args.dataset_root.expanduser(),
        hand_config=args.hand_config.expanduser(),
        hand_port=args.hand_port,
        hand_measure_rate_hz=args.hand_measure_rate_hz,
        episode_time=args.episode_time,
        warmup_time=args.warmup_time,
        active_motors=args.active_motors,
    )
    return run_hand_only(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
