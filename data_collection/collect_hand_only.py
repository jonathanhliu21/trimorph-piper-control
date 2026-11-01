import argparse
import math
import queue
import shutil
import sys
import threading
import time

from dataclasses import dataclass, replace
from pathlib import Path

# Support both direct execution and package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_collection.collection.config import (
    WORKER_INIT_TIMEOUT_S,
    VIDEO_FLUSH_TIMEOUT_S,
    DEFAULT_DEMONSTRATION_DURATION_S,
    DEFAULT_WARMUP_TIME_S,
    DEFAULT_HAND_MEASURE_RATE_HZ,
    DEFAULT_PIPER_RATE_HZ,
    DEFAULT_PIPER_INTERFACE,
    DEFAULT_PIPER_CHANNEL,
    DEFAULT_PIPER_FIRMWARE,
    DEFAULT_PIPER_INSTALLATION_POS,
    DEFAULT_PIPER_PAYLOAD,
    DEFAULT_TEACHER_MODE_POLL_HZ,
    DEFAULT_HAND_CONFIG,
    DEFAULT_HAND_PORT,
    DEFAULT_QUEST_JUMP_THRESHOLD,
    DATA_COLLECTION_MODE,
    RETARGET_ONLY_MODE,
    DEFAULT_THUMB_MODEL,
    HandOnlyCollectionConfig,
    DEFAULT_DATASET_ROOT,
    TASK_VIEW_WINDOW_NAME,
    TRIMORPH_MOTOR_IDS,
    validate_camera_settings,
)
from data_collection.collection.depthai_task_camera_worker import run_depthai_task_camera_worker
from data_collection.collection.episode import (
    EpisodeBuffer,
    RecordingManager,
    episode_paths,
    next_episode_index,
    raw_root,
)
from data_collection.collection.follower_hand_worker import run_follower_hand_worker
from data_collection.collection.leader_hand_worker import run_leader_hand_worker
from data_collection.collection.piper_worker import run_piper_worker
from data_collection.collection.quest_worker import run_quest_worker
from data_collection.collection.mocap_worker import run_mocap_worker
from data_collection.collection.shared_state import SharedState
from data_collection.collection.tactile_worker import run_tactile_worker
from data_collection.collection.utils import WorkerError, import_cv2, worker_wrapper
from scripts.piper_api.foot_pedal import FootPedalTrigger


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
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
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
    parser.add_argument("--use-mocap", action="store_true", help="Record ROS 2 Vicon poses/markers to each episode's mocap.csv.")
    parser.add_argument("--mocap-topic", dest="mocap_topics", action="append", default=[], help="Select a Vicon topic (repeatable); default: discover all supported topics. Requires --use-mocap.")
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
        "--quest-jump-threshold", type=float, default=DEFAULT_QUEST_JUMP_THRESHOLD,
        help="Warn when consecutive Quest packets move farther than this distance in native position units (default: 0.05).",
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
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--piper-start-joints-deg",
        type=float,
        nargs=6,
        metavar="DEG",
        help=(
            "Before collecting, enable Piper and move it in joint space to these "
            "angles, leaving it holding there; its measured flange pose is saved "
            "in every episode's metadata as the postprocessing start pose."
        ),
    )
    start_group.add_argument(
        "--piper-start-home",
        action="store_true",
        help="Like --piper-start-joints-deg, using the teleop home (config/teleop_home.json).",
    )
    parser.add_argument(
        "--piper-start-speed-deg-s",
        type=float,
        default=10.0,
        help="Joint speed of the move to the start pose (default: %(default)s).",
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
            "scripts.thumb_retargeting.fit_model_to_endpoints.py (default: %(default)s)."
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
    if args.mocap_topics and not args.use_mocap:
        parser.error("--mocap-topic requires --use-mocap")
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
    if config.use_mocap:
        specs.append(("mocap", run_mocap_worker))
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
        f"{len(session.quest_rows)} Quest samples, "
        f"{len(session.mocap_rows)} mocap rows"
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
        """Handle pedal start, save on release, and discard after a long hold."""

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

    from data_collection.collection.live_contact_widget import LiveContactWidget

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


def move_piper_to_start(config: HandOnlyCollectionConfig) -> tuple[float, ...]:
    """Enable Piper, ramp it in joint space to the start joints and leave it holding.

    Returns the measured flange pose [x, y, z, roll, pitch, yaw] (m, rad).
    """

    import numpy as np

    from scripts.piper_api.piper_api import PiperAPI
    from scripts.teleop.quest_teleop import POSITION_MODE_SETTLE_S, go_home, home_warnings

    for warning in home_warnings(config.piper_start_joints_deg):
        print(f"WARNING: start pose {warning}")
    api = PiperAPI(
        firmware=config.piper_firmware,
        interface=config.piper_interface,
        channel=config.piper_channel,
        installation_pos=config.piper_installation_pos,
        payload=config.piper_payload,
    )
    api.connect()
    try:
        if not api.is_ok():
            raise RuntimeError("Piper is not responding on the CAN bus")
        input(
            "Clear the workspace, keep a hand near the e-stop, and press Enter to "
            "move Piper to the start pose..."
        )
        if not api.enable():
            raise RuntimeError("Timed out while enabling Piper")
        api.set_position_mode()
        time.sleep(POSITION_MODE_SETTLE_S)
        api.set_speed_percent(100)
        go_home(
            api,
            np.radians(config.piper_start_joints_deg),
            config.piper_start_speed_deg_s,
            50.0,
        )
        pose = tuple(float(value) for value in api.get_eef_pose())
    finally:
        # Disconnecting does not disable the arm: it keeps holding the start pose.
        api.disconnect()
    return pose


def run_hand_only(config: HandOnlyCollectionConfig) -> int:
    shared = SharedState()
    recorder = RecordingManager()
    errors: queue.Queue[WorkerError] = queue.Queue()
    pedal: FootPedalTrigger | None = None
    pedal_controller: FootPedalCollectionController | None = None

    if getattr(config, "piper_start_joints_deg", None) is not None:
        pose = move_piper_to_start(config)
        config = replace(config, piper_start_pose=pose)
        print(
            "Piper is holding the start pose; postprocessing uses it from each "
            "episode's metadata (--piper-start-pose "
            + " ".join(f"{value:.4f}" for value in pose)
            + ")."
        )

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
    piper_start_joints_deg = args.piper_start_joints_deg
    if args.piper_start_home:
        from scripts.teleop.quest_teleop import saved_home_deg

        piper_start_joints_deg = saved_home_deg()
    if piper_start_joints_deg is not None:
        piper_start_joints_deg = tuple(piper_start_joints_deg)
        if not all(math.isfinite(value) for value in piper_start_joints_deg):
            raise ValueError("--piper-start-joints-deg must be finite")
        if not math.isfinite(args.piper_start_speed_deg_s) or args.piper_start_speed_deg_s <= 0:
            raise ValueError("--piper-start-speed-deg-s must be finite and greater than 0")
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
        use_mocap=args.use_mocap,
        mocap_topics=tuple(args.mocap_topics),
        quest_jump_threshold=args.quest_jump_threshold,
        piper_joint_level=args.piper_joint_level,
        piper_rate_hz=args.piper_rate_hz,
        teacher_mode_poll_hz=args.piper_teacher_mode_poll_hz,
        piper_channel=args.piper_channel,
        piper_interface=args.piper_interface,
        piper_firmware=args.piper_firmware,
        piper_installation_pos=args.piper_installation_pos,
        piper_payload=args.piper_payload,
        piper_start_joints_deg=piper_start_joints_deg,
        piper_start_speed_deg_s=args.piper_start_speed_deg_s,
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
        if not math.isfinite(config.quest_jump_threshold) or config.quest_jump_threshold <= 0:
            raise ValueError("--quest-jump-threshold must be finite and greater than 0")
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
