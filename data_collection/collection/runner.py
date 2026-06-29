from __future__ import annotations

import queue
import shutil
import threading
import time

from scripts.piper_api.foot_pedal import FootPedalTrigger

from .config import CollectionConfig
from .episode import (
    EpisodeBuffer,
    RecordingManager,
    episode_paths,
    next_episode_index,
    raw_root,
)
from .curobo_worker import run_curobo_worker
from .midas_worker import run_midas_worker
from .piper_worker import run_piper_worker
from .realsense_worker import run_realsense_worker
from .shared_state import SharedState
from .tactile_worker import run_tactile_worker
from .teleop_worker import run_teleop_worker
from .utils import WorkerError, import_cv2, worker_wrapper


_WORKER_INIT_TIMEOUT_S = 60.0


def start_workers(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    errors: queue.Queue[WorkerError],
) -> list[threading.Thread]:
    from scripts.piper_api.curobo_trajectory_buffer import CuRoboTrajectoryBuffer

    trajectory_buffer = CuRoboTrajectoryBuffer() if config.curobo_enabled else None

    # Bind trajectory_buffer into workers via lambda so worker_wrapper stays unchanged.
    def _piper(cfg, sh, rec, *, init_done):
        run_piper_worker(cfg, sh, rec, trajectory_buffer, init_done=init_done)

    def _curobo(cfg, sh, rec, *, init_done):
        run_curobo_worker(cfg, sh, rec, trajectory_buffer, init_done=init_done)

    specs = [
        ("realsense", run_realsense_worker),
        ("piper", _piper),
        ("midas", run_midas_worker),
        ("teleop", run_teleop_worker),
    ]
    if config.enable_tactile:
        specs.append(("tactile", run_tactile_worker))
    if config.curobo_enabled:
        specs.append(("curobo", _curobo))

    threads = []
    for name, target in specs:
        init_done = threading.Event()
        thread = threading.Thread(
            target=worker_wrapper,
            name=f"{name}-worker",
            args=(name, errors, shared, init_done, target, config, shared, recorder),
            daemon=True,
        )
        print(f"Starting {name} worker...")
        thread.start()
        threads.append(thread)
        init_done.wait(timeout=_WORKER_INIT_TIMEOUT_S)
        if shared.shutdown.is_set():
            break
    return threads


def _rate_hz_from_timestamps_ns(timestamps_ns: list[int]) -> float:
    if len(timestamps_ns) < 2:
        return 0.0
    elapsed_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    if elapsed_s <= 0:
        return 0.0
    return (len(timestamps_ns) - 1) / elapsed_s


def _duration_s_from_timestamps_ns(timestamps_ns: list[int]) -> float:
    if len(timestamps_ns) < 2:
        return 0.0
    return max(0.0, (timestamps_ns[-1] - timestamps_ns[0]) / 1e9)


def _row_timestamps_ns(rows: list[dict[str, object]]) -> list[int]:
    return [int(row["monotonic_ns"]) for row in rows]


def _print_episode_summary(session: EpisodeBuffer) -> None:
    duration_s = (
        ((session.stop_ns or time.monotonic_ns()) - session.start_ns) / 1e9
    )
    summaries = [
        (
            "camera",
            len(session.camera_rows),
            "frames",
            _row_timestamps_ns(session.camera_rows),
        ),
        (
            "Midas",
            len(session.midas_rows),
            "samples",
            _row_timestamps_ns(session.midas_rows),
        ),
        (
            "Piper",
            len(session.piper_rows),
            "samples",
            _row_timestamps_ns(session.piper_rows),
        ),
        (
            "tactile",
            len(session.tactile_timestamps),
            "samples",
            [sample_ns for sample_ns, _, _ in session.tactile_timestamps],
        ),
    ]
    if session.config.record_teleop_camera:
        summaries.append(
            (
                "teleop camera",
                len(session.teleop_camera_rows),
                "frames",
                _row_timestamps_ns(session.teleop_camera_rows),
            )
        )

    print(f"Saved episode {session.paths.episode_index:06d}: duration {duration_s:.3f}s")
    for name, count, unit, timestamps_ns in summaries:
        episode_rate_hz = count / duration_s if duration_s > 0 else 0.0
        active_duration_s = _duration_s_from_timestamps_ns(timestamps_ns)
        active_rate_hz = _rate_hz_from_timestamps_ns(timestamps_ns)
        print(
            f"  {name}: {count} {unit} @ {episode_rate_hz:.2f} Hz over episode "
            f"(active {active_rate_hz:.2f} Hz for {active_duration_s:.3f}s)"
        )


_GREEN = "\033[32m"
_ORANGE = "\033[38;5;214m"
_RESET = "\033[0m"

HOLD_DELETE_S = 2.0


def stop_and_flush_episode(
    recorder: RecordingManager,
    reason: str,
    *,
    timeout_s: float = 5.0,
    color: str = "",
) -> None:
    session = recorder.stop(reason)
    if session is None:
        return
    print(f"Stopping episode {session.paths.episode_index:06d}: {reason}")
    session.video_done.wait(timeout=timeout_s)
    session.teleop_video_done.wait(timeout=timeout_s)
    session.flush()
    _print_episode_summary(session)
    if color:
        print(f"{color}Episode {session.paths.episode_index:06d} saved.{_RESET}")


def delete_and_discard_episode(
    recorder: RecordingManager,
    *,
    timeout_s: float = 5.0,
) -> None:
    session = recorder.stop("foot_pedal_delete")
    if session is None:
        return
    episode_index = session.paths.episode_index
    print(f"Discarding episode {episode_index:06d}...")
    session.video_done.wait(timeout=timeout_s)
    session.teleop_video_done.wait(timeout=timeout_s)
    shutil.rmtree(session.paths.episode_dir, ignore_errors=True)
    print(f"{_ORANGE}Episode {episode_index:06d} deleted.{_RESET}")


def start_episode(config: CollectionConfig, recorder: RecordingManager) -> None:
    episode_index = next_episode_index(raw_root(config.dataset_root))
    paths = episode_paths(config.dataset_root, episode_index)
    paths.episode_dir.mkdir(parents=True, exist_ok=False)
    session = EpisodeBuffer(paths, config)
    recorder.start(session)
    print(f"Recording episode {episode_index:06d}. Press pedal to save, hold {HOLD_DELETE_S:.0f}s to delete.")


def raise_next_worker_error(
    errors: queue.Queue[WorkerError],
    recorder: RecordingManager,
) -> None:
    try:
        error = errors.get_nowait()
    except queue.Empty:
        return

    print(error.traceback_text, end="")
    session = recorder.current()
    if session is not None:
        session.add_error(error)
        stop_and_flush_episode(recorder, f"{error.name}_worker_error")
    raise RuntimeError(f"{error.name} worker failed: {error.message}")


def main_loop(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    pedal: FootPedalTrigger,
    errors: queue.Queue[WorkerError],
) -> None:
    cv2 = import_cv2()
    pending_start_deadline: float | None = None
    next_idle_notice = 0.0
    recording_pedal_down = False

    while True:
        raise_next_worker_error(errors, recorder)
        if shared.shutdown.is_set():
            break

        realsense_bgr, teleop_bgr = shared.display_frames()
        if config.show_realsense_view and realsense_bgr is not None:
            cv2.imshow(config.realsense_window_name, realsense_bgr)
        if config.show_teleop_view and teleop_bgr is not None:
            cv2.imshow(config.teleop_window_name, teleop_bgr)
        key = cv2.waitKey(1) & 0xFF
        teacher_active, teacher_status = shared.teacher_mode()
        if key in (ord("q"), 27):
            shared.shutdown.set()
            break
        if key == ord("c"):
            shared.calibrate_neutral.set()
        if key == ord("e"):
            if teacher_active:
                shared.set_teleop_enabled(False)
                print("Arm teleop disabled while Piper teacher mode is active.")
            else:
                enabled = shared.toggle_teleop_enabled()
                print(f"Arm teleop enabled: {enabled}")
        if (
            recorder.is_recording()
            and config.stop_episode_on_teacher_mode_lost
            and not teacher_active
        ):
            stop_and_flush_episode(recorder, "teacher_mode_lost")
            pending_start_deadline = None
            recording_pedal_down = False
            continue

        requested_stop = shared.pop_stop_episode_reason()
        if recorder.is_recording() and requested_stop == "teacher_mode_lost":
            stop_and_flush_episode(recorder, requested_stop)
            pending_start_deadline = None
            recording_pedal_down = False
            continue

        pedal_pressed = bool(pedal.poll())
        if recorder.is_recording():
            hold_s = pedal.current_hold_s()
            if pedal_pressed and not recording_pedal_down:
                recording_pedal_down = True
                print(f"Pedal held. Release to save, hold {HOLD_DELETE_S:.0f}s to delete.")
            if recording_pedal_down:
                if hold_s is not None and hold_s >= HOLD_DELETE_S:
                    recording_pedal_down = False
                    delete_and_discard_episode(recorder)
                elif hold_s is None:
                    recording_pedal_down = False
                    stop_and_flush_episode(recorder, "foot_pedal", color=_GREEN)
            continue

        if pending_start_deadline is None:
            if pedal_pressed:
                if teacher_active:
                    pending_start_deadline = time.monotonic() + config.ready_seconds
                    print(
                        f"Pedal pressed. Recording starts in {config.ready_seconds:.1f}s."
                    )
                else:
                    print(
                        "Pedal press ignored because Piper teacher mode is inactive "
                        f"(teach_status={teacher_status!r})."
                    )
            else:
                now = time.monotonic()
                if not teacher_active and now >= next_idle_notice:
                    print(
                        "Waiting for Piper teacher mode before teleop/recording "
                        f"(teach_status={teacher_status!r})."
                    )
                    next_idle_notice = now + 2.0
        else:
            if not teacher_active:
                print("Start cancelled because Piper teacher mode is inactive.")
                pending_start_deadline = None
            elif pedal_pressed:
                print("Start cancelled by second pedal press during ready delay.")
                pending_start_deadline = None
            elif time.monotonic() >= pending_start_deadline:
                start_episode(config, recorder)
                pending_start_deadline = None


def run_collection(config: CollectionConfig) -> int:
    config.dataset_root.mkdir(parents=True, exist_ok=True)
    raw_root(config.dataset_root).mkdir(parents=True, exist_ok=True)

    shared = SharedState()
    shared.set_teleop_enabled(config.teleop_enable_default)
    recorder = RecordingManager()
    errors: queue.Queue[WorkerError] = queue.Queue()

    pedal = FootPedalTrigger(
        config.foot_pedal_device,
        key_code=config.foot_pedal_key_code,
        debounce_s=config.foot_pedal_debounce_seconds,
    )
    pedal.start()
    print(f"Foot pedal input: {pedal.active_device}")
    print("Press c in a display window to calibrate teleop neutral. Press q to quit.")

    threads = start_workers(config, shared, recorder, errors)
    deferred_error: WorkerError | None = None
    try:
        main_loop(config, shared, recorder, pedal, errors)
    except KeyboardInterrupt:
        print("\nCollection stopped.")
    except RuntimeError:
        raise
    finally:
        try:
            deferred_error = errors.get_nowait()
        except queue.Empty:
            deferred_error = None
        session = recorder.current()
        if session is not None:
            if deferred_error is not None:
                session.add_error(deferred_error)
            stop_and_flush_episode(recorder, "shutdown")
        shared.shutdown.set()
        pedal.stop()
        for thread in threads:
            thread.join(timeout=3.0)
        try:
            import_cv2().destroyAllWindows()
        except Exception:
            pass
    if deferred_error is not None:
        print(deferred_error.traceback_text, end="")
        raise RuntimeError(
            f"{deferred_error.name} worker failed: {deferred_error.message}"
        )
    return 0
