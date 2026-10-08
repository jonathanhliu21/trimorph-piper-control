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
from collection.tactile_worker import run_tactile_worker
from collection.utils import WorkerError, worker_wrapper

WORKER_INIT_TIMEOUT_S = 60.0
DEFAULT_DEMONSTRATION_DURATION_S = 30.0
DEFAULT_WARMUP_TIME_S = 5.0
DEFAULT_HAND_MEASURE_RATE_HZ = 50.0
DEFAULT_FOLLOWER_MEASURE_RATE_HZ = 100.0
DEFAULT_FORCE_FEEDBACK_RATE_HZ = 50.0
DEFAULT_HAND_CONFIG = Path("config/hand.yaml")
DEFAULT_HAND_PORT = "/dev/serial/by-id/REPLACE_WITH_TRIMORPH_HAND"
FOLLOWER_PROFILE_VELOCITY_RAD_S = 3.0
FOLLOWER_PROFILE_ACCELERATION_RAD_S2 = 50.0
DATA_COLLECTION_MODE = "data_collection"
RETARGET_ONLY_MODE = "retarget_only"
DEFAULT_THUMB_MODEL = Path("thumb_retarget_model.json")

PIP_SCALE = 0.5
MCP_SCALE = 0.5

FORCE_FEEDBACK_MOTOR_IDS = (0, 1, *range(4, 13))
# Map each enabled TRIMORPH thumb joint to its MIDAS contact-error source.
FORCE_FEEDBACK_SOURCE_MOTOR_IDS = {
    0: 0,
    1: 1,
}
FORCE_FEEDBACK_MAX_CURRENT_MA = 100.0
FORCE_FEEDBACK_FULL_SCALE_ERROR_RAD = 0.01
TACTILE_FORCE_FEEDBACK_CURRENT_MA = 50.0
TACTILE_FORCE_FEEDBACK_MOTOR_IDS = (
    (4, 5, 6),
    (7, 8, 9),
    (10, 11, 12),
    (0, 1),
)


@dataclass(frozen=True)
class HandOnlyCollectionConfig:
    dataset_root: Path
    hand_config: Path
    hand_port: str
    hand_measure_rate_hz: float

    warmup_time: float
    episode_time: float
    mode: str = DATA_COLLECTION_MODE
    show_ui: bool = False
    add_tactile: bool = False
    paxini_port: str | None = None
    tactile_debug_log: Path | None = None
    force_feedback_motor_ids: tuple[int, ...] = FORCE_FEEDBACK_MOTOR_IDS
    force_feedback_source_motor_ids: dict[int, int] = field(
        default_factory=lambda: dict(FORCE_FEEDBACK_SOURCE_MOTOR_IDS)
    )
    force_feedback_max_current_ma: float = FORCE_FEEDBACK_MAX_CURRENT_MA
    force_feedback_rate_hz: float = DEFAULT_FORCE_FEEDBACK_RATE_HZ
    force_feedback_full_scale_error_rad: float = (
        FORCE_FEEDBACK_FULL_SCALE_ERROR_RAD
    )
    tactile_force_feedback_current_ma: float = TACTILE_FORCE_FEEDBACK_CURRENT_MA
    tactile_force_feedback_motor_ids: tuple[tuple[int, ...], ...] = (
        TACTILE_FORCE_FEEDBACK_MOTOR_IDS
    )
    active_motors: list[int] = field(
        default_factory=lambda: list(TRIMORPH_MOTOR_IDS)
    )
    follower_hand_config: Path | None = None
    follower_hand_port: str | None = None
    follower_measure_rate_hz: float = DEFAULT_FOLLOWER_MEASURE_RATE_HZ
    follower_command_rate_hz: float = 50.0
    follower_profile_velocity_rad_s: float = FOLLOWER_PROFILE_VELOCITY_RAD_S
    follower_profile_acceleration_rad_s2: float = (
        FOLLOWER_PROFILE_ACCELERATION_RAD_S2
    )
    follower_configure_hardware: bool = True
    follower_command_scale: float = 1.0
    follower_max_step_rad: float = 0.15
    follower_interpolation_alpha: float = 0.4
    thumb_retarget_model: Path | None = None
    follower_joint_command_scales: dict[int, float] = field(
        default_factory=lambda: {
            0: 1.0,
            1: 1.0,
            2: 1.0,
            3: 1.0,
            4: PIP_SCALE,
            5: MCP_SCALE,
            6: 1.0,
            7: PIP_SCALE,
            8: MCP_SCALE,
            9: 1.0,
            10: PIP_SCALE,
            11: MCP_SCALE,
            12: 1.0,
        }
    )
    # MIDAS target, in radians, when the matching TRIMORPH joint is at 0 rad.
    # Joints 2 and 3 are overridden by the learned thumb model when it is used.
    follower_joint_command_offsets: dict[int, float] = field(
        default_factory=lambda: {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0,
            6: 0.0,
            7: 0.0,
            8: 0.0,
            9: 0.0,
            10: 0.0,
            11: 0.0,
            12: 0.262,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget a follower from the TRIMORPH leader hand, optionally recording "
            "a fixed-duration episode."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--retarget-only",
        dest="mode",
        action="store_const",
        const=RETARGET_ONLY_MODE,
        help="Retarget indefinitely without recording; stop with Ctrl-C.",
    )
    mode_group.add_argument(
        "--data-collection",
        dest="mode",
        action="store_const",
        const=DATA_COLLECTION_MODE,
        help="Record one fixed-duration episode (the default).",
    )
    parser.set_defaults(mode=DATA_COLLECTION_MODE)
    parser.add_argument("--dataset-root", type=Path, default=Path("data_trimorph"))
    parser.add_argument("--hand-config", type=Path, default=DEFAULT_HAND_CONFIG)
    parser.add_argument(
        "--hand-port", default=DEFAULT_HAND_PORT
    )
    parser.add_argument("--follower-config", type=Path)
    parser.add_argument("--follower-port")
    parser.add_argument(
        "--show-ui",
        action="store_true",
        help="Show the read-only follower MIDAS live-contact visualization.",
    )
    parser.add_argument(
        "--add-tactile",
        action="store_true",
        help=(
            "connect the auto-detected Paxini tactile sensor and show binary "
            "fingertip contacts in the live-contact UI"
        ),
    )
    parser.add_argument(
        "--paxini-port",
        metavar="PORT",
        help=(
            "Paxini serial port, for example /dev/cu.usbserial-XXXX; providing "
            "this option also enables tactile sensing"
        ),
    )
    parser.add_argument(
        "--tactile-debug-log",
        type=Path,
        metavar="PATH",
        help=(
            "write structured tactile contact-detector diagnostics to PATH; "
            "providing this option also enables tactile sensing"
        ),
    )
    parser.add_argument(
        "--thumb-model",
        type=Path,
        default=DEFAULT_THUMB_MODEL,
        help=(
            "Linear TRIMORPH-to-MIDAS thumb model produced by "
            "fit_model_to_endpoints.py (default: %(default)s)."
        ),
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
        help=(
            "Motor IDs to read, command, and record. The hand and follower YAML "
            "files may contain additional motor IDs."
        ),
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
    if config.add_tactile:
        specs.append(("tactile", run_tactile_worker))

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


def run_follower_contact_ui(
    shared: SharedState,
    *,
    duration_s: float | None,
) -> str:
    """Run the read-only follower contact view until close or shutdown."""

    import tkinter as tk

    from collection.live_contact_widget import LiveContactWidget

    root = tk.Tk()
    root.title("Follower hand contact")
    root.minsize(1000, 500)
    widget = LiveContactWidget(root)
    widget.pack(fill="both", expand=True, padx=12, pady=12)
    deadline = None if duration_s is None else time.monotonic() + duration_s
    exit_reason = "ui_closed"

    def close_window() -> None:
        nonlocal exit_reason
        exit_reason = "ui_closed"
        shared.shutdown.set()

    def refresh() -> None:
        nonlocal exit_reason
        snapshot = shared.latest_follower_contact_snapshot()
        tactile_contacts = shared.latest_tactile_contacts()
        if snapshot is not None:
            widget.show_by_motor_id(
                snapshot.motor_ids,
                snapshot.contact_feedback,
                snapshot.normalized_joint_difference,
                tactile_contacts=tactile_contacts,
            )
        elif tactile_contacts is not None:
            widget.show(tactile_contacts=tactile_contacts)
        if deadline is not None and time.monotonic() >= deadline:
            exit_reason = "duration_complete"
            shared.shutdown.set()
        if shared.shutdown.is_set():
            root.destroy()
            return
        root.after(30, refresh)

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.after(0, refresh)
    try:
        root.mainloop()
    finally:
        shared.shutdown.set()
    return exit_reason


def run_hand_only(config: HandOnlyCollectionConfig) -> int:
    shared = SharedState()
    recorder = RecordingManager()
    errors: queue.Queue[WorkerError] = queue.Queue()

    if config.mode == DATA_COLLECTION_MODE:
        print(f"Warming up for {config.warmup_time}s")
        time.sleep(config.warmup_time)
        print("Warmup done, starting data collection...")
    elif config.mode != RETARGET_ONLY_MODE:
        raise ValueError(f"Unknown hand-only mode: {config.mode!r}")

    threads = initialize_worker(config, shared, recorder, errors)
    # Retain compatibility with callers that replace the initializer in tests.
    if isinstance(threads, threading.Thread) or hasattr(threads, "join"):
        threads = [threads]
    reason = "duration_complete"
    try:
        if config.mode == RETARGET_ONLY_MODE:
            print("Retarget-only mode active. Press Ctrl-C to stop.")
            if config.show_ui:
                reason = run_follower_contact_ui(shared, duration_s=None)
            else:
                shared.shutdown.wait()
        else:
            start_episode(config, recorder)
            if config.show_ui:
                reason = run_follower_contact_ui(
                    shared,
                    duration_s=config.episode_time,
                )
            else:
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
        if config.mode == RETARGET_ONLY_MODE:
            print("\nRetargeting stopped.")
        else:
            print("\nCollection interrupted; saving recorded samples.")
    finally:
        shared.shutdown.set()
        for thread in threads:
            thread.join(timeout=3.0)
        if config.mode == DATA_COLLECTION_MODE:
            stop_episode(recorder, reason)
    return 0


def main() -> int:
    args = parse_args()
    add_tactile = (
        args.add_tactile
        or args.paxini_port is not None
        or args.tactile_debug_log is not None
    )
    config = HandOnlyCollectionConfig(
        dataset_root=args.dataset_root.expanduser(),
        hand_config=args.hand_config.expanduser(),
        hand_port=args.hand_port,
        hand_measure_rate_hz=DEFAULT_HAND_MEASURE_RATE_HZ,
        episode_time=args.episode_time,
        warmup_time=args.warmup_time,
        mode=args.mode,
        show_ui=args.show_ui,
        add_tactile=add_tactile,
        paxini_port=args.paxini_port,
        tactile_debug_log=(
            args.tactile_debug_log.expanduser()
            if args.tactile_debug_log is not None
            else None
        ),
        active_motors=args.active_motors,
        follower_hand_config=(
            args.follower_config.expanduser()
            if args.follower_config is not None
            else None
        ),
        follower_hand_port=args.follower_port,
        thumb_retarget_model=(
            args.thumb_model.expanduser()
            if args.follower_config is not None
            else None
        ),
    )
    if (config.follower_hand_config is None) != (config.follower_hand_port is None):
        raise ValueError("--follower-config and --follower-port must be provided together")
    if config.show_ui and config.follower_hand_config is None:
        raise ValueError("--show-ui requires --follower-config and --follower-port")
    if config.add_tactile and not config.show_ui and config.tactile_debug_log is None:
        raise ValueError(
            "Tactile sensing requires --show-ui or --tactile-debug-log"
        )
    if config.follower_hand_port == config.hand_port:
        raise ValueError("Primary and follower hand ports must be different")
    if config.paxini_port is not None and config.paxini_port in {
        config.hand_port,
        config.follower_hand_port,
    }:
        raise ValueError("The Paxini and motor serial ports must be different")
    return run_hand_only(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
