from __future__ import annotations

import queue
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
from .midas_worker import run_midas_worker
from .piper_worker import run_piper_worker
from .realsense_worker import run_realsense_worker
from .shared_state import SharedState
from .tactile_worker import run_tactile_worker
from .teleop_worker import run_teleop_worker
from .utils import WorkerError, import_cv2, worker_wrapper


def start_workers(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    errors: queue.Queue[WorkerError],
) -> list[threading.Thread]:
    specs = [
        ("realsense", run_realsense_worker),
        ("piper", run_piper_worker),
        ("midas", run_midas_worker),
        ("teleop", run_teleop_worker),
    ]
    if config.enable_tactile:
        specs.append(("tactile", run_tactile_worker))

    threads = []
    for name, target in specs:
        thread = threading.Thread(
            target=worker_wrapper,
            name=f"{name}-worker",
            args=(name, errors, shared, target, config, shared, recorder),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    return threads


def stop_and_flush_episode(
    recorder: RecordingManager,
    reason: str,
    *,
    timeout_s: float = 5.0,
) -> None:
    session = recorder.stop(reason)
    if session is None:
        return
    print(f"Stopping episode {session.paths.episode_index:06d}: {reason}")
    session.video_done.wait(timeout=timeout_s)
    session.teleop_video_done.wait(timeout=timeout_s)
    session.flush()
    print(
        f"Saved episode {session.paths.episode_index:06d}: "
        f"{len(session.camera_rows)} frames, "
        f"{len(session.midas_rows)} Midas samples, "
        f"{len(session.piper_rows)} Piper samples, "
        f"{len(session.tactile_timestamps)} tactile samples."
    )


def start_episode(config: CollectionConfig, recorder: RecordingManager) -> None:
    episode_index = next_episode_index(raw_root(config.dataset_root))
    paths = episode_paths(config.dataset_root, episode_index)
    paths.episode_dir.mkdir(parents=True, exist_ok=False)
    session = EpisodeBuffer(paths, config)
    recorder.start(session)
    print(f"Recording episode {episode_index:06d}. Press the foot pedal again to stop.")


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

    while not shared.shutdown.is_set():
        try:
            error = errors.get_nowait()
        except queue.Empty:
            error = None
        if error is not None:
            session = recorder.current()
            if session is not None:
                session.add_error(error)
                stop_and_flush_episode(recorder, f"{error.name}_worker_error")
            raise RuntimeError(f"{error.name} worker failed: {error.message}")

        realsense_bgr, teleop_bgr = shared.display_frames()
        if config.show_realsense_view and realsense_bgr is not None:
            cv2.imshow(config.realsense_window_name, realsense_bgr)
        if config.show_teleop_view and teleop_bgr is not None:
            cv2.imshow(config.teleop_window_name, teleop_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            shared.shutdown.set()
            break
        if key == ord("c"):
            shared.calibrate_neutral.set()

        teacher_active, teacher_status = shared.teacher_mode()
        if (
            recorder.is_recording()
            and config.stop_episode_on_teacher_mode_lost
            and not teacher_active
        ):
            stop_and_flush_episode(recorder, "teacher_mode_lost")
            pending_start_deadline = None
            continue

        requested_stop = shared.pop_stop_episode_reason()
        if recorder.is_recording() and requested_stop == "teacher_mode_lost":
            stop_and_flush_episode(recorder, requested_stop)
            pending_start_deadline = None
            continue

        pedal_pressed = bool(pedal.poll())
        if recorder.is_recording():
            if pedal_pressed:
                stop_and_flush_episode(recorder, "foot_pedal")
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
    try:
        main_loop(config, shared, recorder, pedal, errors)
    except KeyboardInterrupt:
        print("\nCollection stopped.")
    finally:
        session = recorder.current()
        if session is not None:
            stop_and_flush_episode(recorder, "shutdown")
        shared.shutdown.set()
        pedal.stop()
        for thread in threads:
            thread.join(timeout=3.0)
        try:
            import_cv2().destroyAllWindows()
        except Exception:
            pass
    return 0
