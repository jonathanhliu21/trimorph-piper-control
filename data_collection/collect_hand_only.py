import argparse
import queue
import shutil
import sys
import threading
import time

from dataclasses import dataclass, field
from pathlib import Path

from collection.config import (
    DEFAULT_QUEST_HOST,
    DEFAULT_QUEST_PORT,
    DEFAULT_QUEST_STARTUP_TIMEOUT_S,
    TASK_CAMERA_ANTI_BANDING_HZ,
    TASK_CAMERA_BRIGHTNESS,
    TASK_CAMERA_FLIP,
    TASK_CAMERA_FPS,
    TASK_CAMERA_HEIGHT,
    TASK_CAMERA_WIDTH,
    TASK_VIEW_WINDOW_NAME,
    TRIMORPH_MOTOR_IDS,
    VIDEO_CODEC,
    validate_camera_settings,
)
from collection.depthai_task_camera_worker import run_depthai_task_camera_worker
from collection.episode import (
    EpisodeBuffer,
    RecordingManager,
    episode_paths,
    next_episode_index,
    raw_root,
)
from collection.follower_hand_worker import run_follower_hand_worker
from collection.leader_hand_worker import run_leader_hand_worker
from collection.piper_worker import run_piper_worker
from collection.quest_worker import run_quest_worker
from collection.shared_state import SharedState
from collection.tactile_worker import run_tactile_worker
from collection.utils import WorkerError, import_cv2, worker_wrapper
from scripts.piper_api.foot_pedal import FootPedalTrigger

WORKER_INIT_TIMEOUT_S = 60.0
VIDEO_FLUSH_TIMEOUT_S = 5.0
DEFAULT_DEMONSTRATION_DURATION_S = 30.0
DEFAULT_WARMUP_TIME_S = 5.0
DEFAULT_HAND_MEASURE_RATE_HZ = 50.0
DEFAULT_FOLLOWER_MEASURE_RATE_HZ = 100.0
DEFAULT_FORCE_FEEDBACK_RATE_HZ = 50.0
DEFAULT_PIPER_RATE_HZ = 250.0
DEFAULT_PIPER_INTERFACE = "socketcan"
DEFAULT_PIPER_CHANNEL = "can0"
DEFAULT_PIPER_FIRMWARE = "v183"
DEFAULT_PIPER_INSTALLATION_POS = "right"
DEFAULT_PIPER_PAYLOAD = "half"
DEFAULT_TEACHER_MODE_POLL_HZ = 20.0
DEFAULT_HOLD_DELETE_S = 2.0
DEFAULT_READY_SECONDS = 0.2
DEFAULT_FOOT_PEDAL_DEBOUNCE_S = 0.5
DEFAULT_HAND_CONFIG = Path("config/hand.yaml")
DEFAULT_HAND_PORT = "/dev/serial/by-id/REPLACE_WITH_TRIMORPH_HAND"
FOLLOWER_PROFILE_VELOCITY_RAD_S = 3.0
FOLLOWER_PROFILE_ACCELERATION_RAD_S2 = 50.0
DATA_COLLECTION_MODE = "data_collection"
RETARGET_ONLY_MODE = "retarget_only"
DEFAULT_THUMB_MODEL = Path("thumb_retarget_model.json")

PIP_SCALE = 0.6
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
    use_foot_pedal: bool = False
    hold_delete_s: float = DEFAULT_HOLD_DELETE_S
    ready_seconds: float = DEFAULT_READY_SECONDS
    foot_pedal_device: str | None = None
    foot_pedal_key_code: int | None = None
    foot_pedal_debounce_s: float = DEFAULT_FOOT_PEDAL_DEBOUNCE_S
    mode: str = DATA_COLLECTION_MODE
    show_ui: bool = False
    add_camera: bool = False
    use_piper: bool = False
    use_quest: bool = False
    quest_host: str = DEFAULT_QUEST_HOST
    quest_port: int = DEFAULT_QUEST_PORT
    quest_startup_timeout_s: float = DEFAULT_QUEST_STARTUP_TIMEOUT_S
    piper_joint_level: bool = False
    piper_rate_hz: float = DEFAULT_PIPER_RATE_HZ
    piper_interface: str = DEFAULT_PIPER_INTERFACE
    piper_channel: str = DEFAULT_PIPER_CHANNEL
    piper_firmware: str = DEFAULT_PIPER_FIRMWARE
    piper_installation_pos: str | None = DEFAULT_PIPER_INSTALLATION_POS
    piper_payload: str | None = DEFAULT_PIPER_PAYLOAD
    teacher_mode_poll_hz: float = DEFAULT_TEACHER_MODE_POLL_HZ
    task_view_window_name: str = TASK_VIEW_WINDOW_NAME
    task_camera_width: int = TASK_CAMERA_WIDTH
    task_camera_height: int = TASK_CAMERA_HEIGHT
    task_camera_fps: int = TASK_CAMERA_FPS
    task_camera_flip: str = TASK_CAMERA_FLIP
    task_camera_brightness: int = TASK_CAMERA_BRIGHTNESS
    task_camera_anti_banding_hz: str = TASK_CAMERA_ANTI_BANDING_HZ
    video_codec: str = VIDEO_CODEC
    add_tactile: bool = False
    paxini_port: str | None = None
    tactile_debug_log: Path | None = None
    force_feedback_motor_ids: tuple[int, ...] = FORCE_FEEDBACK_MOTOR_IDS
    disabled_force_feedback_motor_ids: tuple[int, ...] = ()
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
            "episodes with a foot pedal or for a fixed duration."
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
        help=(
            "Record episodes (the default; foot-pedal controlled unless "
            "--episode-s is set)."
        ),
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
        "--add-camera",
        action="store_true",
        help="record the DepthAI wrist camera and show its live feed",
    )
    pose_source_group = parser.add_mutually_exclusive_group()
    pose_source_group.add_argument(
        "--use-piper",
        action="store_true",
        help=(
            "passively record Piper EEF and drag-teach state alongside the hand; "
            "the collector never commands or enables the arm"
        ),
    )
    pose_source_group.add_argument(
        "--use-quest",
        action="store_true",
        help=(
            "record raw Quest position and quaternion packets received over UDP "
            "alongside the hand"
        ),
    )
    parser.add_argument(
        "--piper-joint-level", "--joint-level", "--joint-position",
        dest="piper_joint_level",
        action="store_true",
        help="Record measured Piper joint angles alongside EEF poses for joint-space replay (still passive).",
    )
    parser.add_argument(
        "--piper-rate-hz",
        type=float,
        default=DEFAULT_PIPER_RATE_HZ,
        help="Piper sampling rate in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--piper-teacher-mode-poll-hz",
        type=float,
        default=DEFAULT_TEACHER_MODE_POLL_HZ,
        help="Piper status polling rate in Hz (default: %(default)s).",
    )
    parser.add_argument("--piper-channel", default=DEFAULT_PIPER_CHANNEL)
    parser.add_argument("--piper-interface", default=DEFAULT_PIPER_INTERFACE)
    parser.add_argument(
        "--piper-firmware",
        choices=("default", "v183", "v188"),
        default=DEFAULT_PIPER_FIRMWARE,
    )
    parser.add_argument(
        "--piper-installation-pos",
        choices=("unchanged", "horizontal", "left", "right"),
        default=DEFAULT_PIPER_INSTALLATION_POS,
    )
    parser.add_argument(
        "--piper-payload",
        choices=("unchanged", "empty", "half", "full"),
        default=DEFAULT_PIPER_PAYLOAD,
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
    episode_group = parser.add_mutually_exclusive_group()
    episode_group.add_argument(
        "--episode-s",
        "--episode-time",
        dest="episode_time",
        type=float,
        default=None,
        help=(
            "record one fixed-duration episode instead of using the foot pedal "
            "(the --episode-time spelling is retained for compatibility)"
        ),
    )
    episode_group.add_argument(
        "--use-foot-pedal",
        action="store_true",
        default=True,
        help=(
            "use the foot pedal to start and save episodes, or discard an episode "
            "with a long hold (default)"
        ),
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
    parser.add_argument(
        "--disable-force-feedback-motors",
        nargs="+",
        type=int,
        default=[],
        metavar="MOTOR_ID",
        help=(
            "Keep these active TRIMORPH motors torque-disabled while retaining "
            "their position readings and retargeting (for example: 8 or 7 8 9)."
        ),
    )
    args = parser.parse_args()
    if args.episode_time is not None:
        args.use_foot_pedal = False
    return args


def worker_specs(config: HandOnlyCollectionConfig):
    specs = []
    if config.add_camera:
        specs.append(("task_camera", run_depthai_task_camera_worker))
    if config.use_piper:
        specs.append(("piper", run_piper_worker))
    if config.use_quest:
        specs.append(("quest", run_quest_worker))
    specs.append(("leader_hand", run_leader_hand_worker))
    if config.follower_hand_config is not None:
        specs.append(("follower", run_follower_hand_worker))
    if config.add_tactile:
        specs.append(("tactile", run_tactile_worker))
    return specs


def initialize_worker(
    config: HandOnlyCollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    errors: queue.Queue[WorkerError],
) -> list[threading.Thread]:
    specs = worker_specs(config)

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
    if config.use_foot_pedal:
        print(
            f"Recording episode {index:06d}. Press pedal to save; "
            f"hold {config.hold_delete_s:g}s to delete."
        )
    else:
        print(f"Recording episode {index:06d} for {config.episode_time:.1f}s...")
    return session


def stop_episode(recorder: RecordingManager, reason: str) -> EpisodeBuffer | None:
    session = recorder.stop(reason)
    if session is None:
        return None
    if session.config.add_camera and not session.video_done.wait(
        timeout=VIDEO_FLUSH_TIMEOUT_S
    ):
        print(
            f"Warning: camera writer did not finish within {VIDEO_FLUSH_TIMEOUT_S:.1f}s"
        )
    session.flush()
    print(
        f"Saved episode {session.paths.episode_index:06d}: "
        f"{len(session.leader_hand_rows)} leader-hand samples, "
        f"{len(session.follower_rows)} follower samples, "
        f"{len(session.camera_rows)} camera frames, "
        f"{len(session.piper_rows)} Piper samples, "
        f"{len(session.quest_rows)} Quest samples"
    )
    return session


def delete_episode(recorder: RecordingManager) -> EpisodeBuffer | None:
    """Stop and discard the current episode after a long pedal hold."""

    session = recorder.stop("foot_pedal_delete")
    if session is None:
        return None
    if session.config.add_camera:
        session.video_done.wait(timeout=VIDEO_FLUSH_TIMEOUT_S)
    shutil.rmtree(session.paths.episode_dir, ignore_errors=True)
    print(f"Episode {session.paths.episode_index:06d} deleted.")
    return session


@dataclass
class FootPedalCollectionController:
    config: HandOnlyCollectionConfig
    recorder: RecordingManager
    pedal: FootPedalTrigger
    pending_start_deadline: float | None = None
    recording_pedal_down: bool = False

    def poll(self) -> None:
        """Apply the same start/save/long-hold-delete behavior as runner.py."""

        pedal_pressed = bool(self.pedal.poll())
        if self.recorder.is_recording():
            hold_s = self.pedal.current_hold_s()
            if pedal_pressed and not self.recording_pedal_down:
                self.recording_pedal_down = True
                print(
                    "Pedal held. Release to save; "
                    f"hold {self.config.hold_delete_s:g}s to delete."
                )
            if self.recording_pedal_down:
                if hold_s is not None and hold_s >= self.config.hold_delete_s:
                    self.recording_pedal_down = False
                    delete_episode(self.recorder)
                elif hold_s is None:
                    self.recording_pedal_down = False
                    stop_episode(self.recorder, "foot_pedal")
            return

        if self.pending_start_deadline is None:
            if pedal_pressed:
                self.pending_start_deadline = (
                    time.monotonic() + self.config.ready_seconds
                )
                print(
                    "Pedal pressed. Recording starts in "
                    f"{self.config.ready_seconds:.1f}s."
                )
        elif pedal_pressed:
            print("Start cancelled by second pedal press during ready delay.")
            self.pending_start_deadline = None
        elif time.monotonic() >= self.pending_start_deadline:
            start_episode(self.config, self.recorder)
            self.pending_start_deadline = None


def run_follower_contact_ui(
    shared: SharedState,
    *,
    duration_s: float | None,
    show_camera: bool = False,
    camera_window_name: str = TASK_VIEW_WINDOW_NAME,
    on_tick=None,
) -> str:
    """Run the read-only follower contact view until close or shutdown."""

    import tkinter as tk

    from collection.live_contact_widget import LiveContactWidget

    root = tk.Tk()
    root.title("Follower hand contact")
    root.minsize(1000, 500)
    widget = LiveContactWidget(root)
    widget.pack(fill="both", expand=True, padx=12, pady=12)
    cv2 = import_cv2() if show_camera else None
    deadline = None if duration_s is None else time.monotonic() + duration_s
    exit_reason = "ui_closed"

    def close_window() -> None:
        nonlocal exit_reason
        exit_reason = "ui_closed"
        shared.shutdown.set()

    def refresh() -> None:
        nonlocal exit_reason
        if on_tick is not None:
            on_tick()
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
        if cv2 is not None:
            frame = shared.latest_task_frame()
            if frame is not None:
                cv2.imshow(camera_window_name, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                exit_reason = "camera_view_closed"
                shared.shutdown.set()
        if shared.shutdown.is_set():
            root.destroy()
            return
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
        if cv2 is not None:
            try:
                cv2.destroyWindow(camera_window_name)
            except Exception:
                pass
    return exit_reason


def run_camera_view(
    shared: SharedState,
    *,
    duration_s: float | None,
    window_name: str,
    on_tick=None,
) -> str:
    """Show camera frames until the duration expires, shutdown, or q/Escape."""

    cv2 = import_cv2()
    deadline = None if duration_s is None else time.monotonic() + duration_s
    reason = "duration_complete"
    try:
        while not shared.shutdown.is_set():
            if on_tick is not None:
                on_tick()
            frame = shared.latest_task_frame()
            if frame is not None:
                cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                reason = "camera_view_closed"
                shared.shutdown.set()
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.005)
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
    return reason


def run_hand_only(config: HandOnlyCollectionConfig) -> int:
    shared = SharedState()
    recorder = RecordingManager()
    errors: queue.Queue[WorkerError] = queue.Queue()
    pedal: FootPedalTrigger | None = None
    pedal_controller: FootPedalCollectionController | None = None

    if config.mode == DATA_COLLECTION_MODE:
        print(f"Warming up for {config.warmup_time}s")
        time.sleep(config.warmup_time)
        print("Warmup done, starting data collection...")
    elif config.mode != RETARGET_ONLY_MODE:
        raise ValueError(f"Unknown hand-only mode: {config.mode!r}")

    if config.use_piper:
        print(
            "Piper logging is passive: use the arm's built-in drag-teach control."
        )
    if config.use_quest:
        print(
            f"Waiting for Quest UDP poses on {config.quest_host}:"
            f"{config.quest_port}."
        )

    if config.use_foot_pedal and config.mode == DATA_COLLECTION_MODE:
        pedal = FootPedalTrigger(
            config.foot_pedal_device,
            key_code=config.foot_pedal_key_code,
            debounce_s=config.foot_pedal_debounce_s,
        )
        pedal.start()
        print(f"Foot pedal input: {pedal.active_device}")
        pedal_controller = FootPedalCollectionController(config, recorder, pedal)

    try:
        threads = initialize_worker(config, shared, recorder, errors)
    except Exception:
        if pedal is not None:
            pedal.stop()
        raise
    # Retain compatibility with callers that replace the initializer in tests.
    if isinstance(threads, threading.Thread) or hasattr(threads, "join"):
        threads = [threads]
    reason = "duration_complete"
    try:
        if config.mode == RETARGET_ONLY_MODE:
            print("Retarget-only mode active. Press Ctrl-C to stop.")
            if config.show_ui:
                if config.add_camera:
                    reason = run_follower_contact_ui(
                        shared,
                        duration_s=None,
                        show_camera=True,
                        camera_window_name=config.task_view_window_name,
                    )
                else:
                    reason = run_follower_contact_ui(shared, duration_s=None)
            elif config.add_camera:
                reason = run_camera_view(
                    shared,
                    duration_s=None,
                    window_name=config.task_view_window_name,
                )
            else:
                shared.shutdown.wait()
        elif config.use_foot_pedal:
            print("Foot-pedal collection active. Press q or Ctrl-C to stop.")
            if config.show_ui:
                reason = run_follower_contact_ui(
                    shared,
                    duration_s=None,
                    show_camera=config.add_camera,
                    camera_window_name=config.task_view_window_name,
                    on_tick=pedal_controller.poll,
                )
            elif config.add_camera:
                reason = run_camera_view(
                    shared,
                    duration_s=None,
                    window_name=config.task_view_window_name,
                    on_tick=pedal_controller.poll,
                )
            else:
                while not shared.shutdown.wait(0.005):
                    pedal_controller.poll()
        else:
            start_episode(config, recorder)
            if config.show_ui:
                if config.add_camera:
                    reason = run_follower_contact_ui(
                        shared,
                        duration_s=config.episode_time,
                        show_camera=True,
                        camera_window_name=config.task_view_window_name,
                    )
                else:
                    reason = run_follower_contact_ui(
                        shared,
                        duration_s=config.episode_time,
                    )
            elif config.add_camera:
                reason = run_camera_view(
                    shared,
                    duration_s=config.episode_time,
                    window_name=config.task_view_window_name,
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
        if pedal is not None:
            pedal.stop()
        for thread in threads:
            thread.join(timeout=3.0)
        if config.mode == DATA_COLLECTION_MODE:
            stop_episode(recorder, reason)
    return 0


def main() -> int:
    args = parse_args()
    disabled_force_feedback_motor_ids = tuple(
        dict.fromkeys(args.disable_force_feedback_motors)
    )
    inactive_disabled_ids = sorted(
        set(disabled_force_feedback_motor_ids) - set(args.active_motors)
    )
    if inactive_disabled_ids:
        raise ValueError(
            "--disable-force-feedback-motors contains IDs that are not active: "
            f"{inactive_disabled_ids}"
        )
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
        episode_time=(
            args.episode_time
            if args.episode_time is not None
            else DEFAULT_DEMONSTRATION_DURATION_S
        ),
        use_foot_pedal=args.use_foot_pedal,
        warmup_time=args.warmup_time,
        mode=args.mode,
        show_ui=args.show_ui,
        add_camera=args.add_camera,
        use_piper=args.use_piper,
        use_quest=args.use_quest,
        piper_joint_level=args.piper_joint_level,
        piper_rate_hz=args.piper_rate_hz,
        teacher_mode_poll_hz=args.piper_teacher_mode_poll_hz,
        piper_channel=args.piper_channel,
        piper_interface=args.piper_interface,
        piper_firmware=args.piper_firmware,
        piper_installation_pos=args.piper_installation_pos,
        piper_payload=args.piper_payload,
        add_tactile=add_tactile,
        paxini_port=args.paxini_port,
        tactile_debug_log=(
            args.tactile_debug_log.expanduser()
            if args.tactile_debug_log is not None
            else None
        ),
        active_motors=args.active_motors,
        disabled_force_feedback_motor_ids=disabled_force_feedback_motor_ids,
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
    if config.add_camera:
        validate_camera_settings(config)
    if config.piper_joint_level and not config.use_piper:
        raise ValueError("--piper-joint-level requires --use-piper")
    if config.use_piper:
        if config.piper_rate_hz <= 0:
            raise ValueError("PIPER_RATE_HZ must be greater than 0")
        if config.teacher_mode_poll_hz <= 0:
            raise ValueError("TEACHER_MODE_POLL_HZ must be greater than 0")
        if config.piper_firmware not in {"default", "v183", "v188"}:
            raise ValueError("PIPER_FIRMWARE must be one of: default, v183, v188")
    if config.use_quest:
        if not config.quest_host:
            raise ValueError("QUEST_HOST must not be empty")
        if not 1 <= config.quest_port <= 65535:
            raise ValueError("QUEST_PORT must be in range 1..65535")
        if config.quest_startup_timeout_s <= 0:
            raise ValueError("QUEST_STARTUP_TIMEOUT_S must be greater than 0")
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
