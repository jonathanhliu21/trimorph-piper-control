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
from collection.follower_hand_worker import run_follower_hand_worker
from collection.leader_hand_worker import run_leader_hand_worker
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
    follower_hand_config: Path | None = None
    follower_hand_port: str | None = None
    follower_measure_rate_hz: float = 200.0
    follower_command_rate_hz: float = 50.0
    follower_configure_hardware: bool = True
    follower_command_scale: float = 1.0
    follower_max_step_rad: float = 0.15
    follower_interpolation_alpha: float = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record one fixed-duration TRIMORPH leader-hand episode."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data_trimorph"))
    parser.add_argument("--hand-config", type=Path, default=Path("config/hand.yaml"))
    parser.add_argument(
        "--hand-port", default="/dev/serial/by-id/REPLACE_WITH_TRIMORPH_HAND"
    )
    parser.add_argument("--follower-config", type=Path)
    parser.add_argument("--follower-port")
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
) -> list[threading.Thread]:
    specs = [("leader_hand", run_leader_hand_worker)]
    if config.follower_hand_config is not None:
        specs.append(("follower", run_follower_hand_worker))

    threads: list[threading.Thread] = []
    try:
        for name, target in specs:
            init_done = threading.Event()
            thread = threading.Thread(
                target=worker_wrapper,
                name=f"{name}-worker",
                args=(
                    name,
                    errors,
                    shared,
                    init_done,
                    target,
                    config,
                    shared,
                    recorder,
                ),
                daemon=True,
            )
            print(f"Starting {name} worker...")
            thread.start()
            threads.append(thread)
            if not init_done.wait(timeout=WORKER_INIT_TIMEOUT_S):
                raise TimeoutError(
                    f"{name.capitalize()} worker did not initialize within "
                    f"{WORKER_INIT_TIMEOUT_S}s"
                )
            if not errors.empty():
                error = errors.get_nowait()
                raise RuntimeError(f"{error.name} worker failed: {error.message}")
    except Exception:
        shared.shutdown.set()
        for thread in threads:
            thread.join(timeout=3.0)
        raise

    return threads


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
        f"{len(session.leader_hand_rows)} leader-hand samples, "
        f"{len(session.follower_rows)} follower samples"
    )
    return session


def run_hand_only(config: HandOnlyCollectionConfig) -> int:
    shared = SharedState()
    recorder = RecordingManager()
    errors: queue.Queue[WorkerError] = queue.Queue()

    print(f"Warming up for {config.warmup_time}s")
    time.sleep(config.warmup_time)
    print("Warmup done, starting data collection...")

    threads = initialize_worker(config, shared, recorder, errors)
    # Retain compatibility with callers that replace the initializer in tests.
    if isinstance(threads, threading.Thread) or hasattr(threads, "join"):
        threads = [threads]
    reason = "duration_complete"
    try:
        start_episode(config, recorder)
        shared.shutdown.wait(config.episode_time)
        if not errors.empty():
            error = errors.get_nowait()
            session = recorder.current()
            if session is not None:
                session.add_error(error)
            reason = f"{error.name}_worker_error"
            raise RuntimeError(f"{error.name} worker failed: {error.message}")
    except KeyboardInterrupt:
        reason = "keyboard_interrupt"
        print("\nCollection interrupted; saving recorded samples.")
    finally:
        shared.shutdown.set()
        for thread in threads:
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
        follower_hand_config=(
            args.follower_config.expanduser()
            if args.follower_config is not None
            else None
        ),
        follower_hand_port=args.follower_port,
    )
    if (config.follower_hand_config is None) != (config.follower_hand_port is None):
        raise ValueError("--follower-config and --follower-port must be provided together")
    if config.follower_hand_port == config.hand_port:
        raise ValueError("Primary and follower hand ports must be different")
    return run_hand_only(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
