from __future__ import annotations

import queue
import threading
import time

from scripts.piper_api.foot_pedal import FootPedalTrigger

from .config import CollectionConfig
from .depthai_task_camera_worker import run_depthai_task_camera_worker
from .episode import RecordingManager, raw_root
from .midas_worker import run_midas_worker
from .piper_worker_basic import run_piper_worker_basic
from .runner import (
    HOLD_DELETE_S,
    delete_and_discard_episode,
    raise_next_worker_error,
    start_episode,
    stop_and_flush_episode,
)
from .shared_state import SharedState
from .teleop_worker import run_teleop_worker
from .utils import WorkerError, import_cv2, worker_wrapper

_WORKER_INIT_TIMEOUT_S = 60.0
_GREEN = "\033[32m"


def start_workers_basic(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    errors: "queue.Queue[WorkerError]",
) -> list[threading.Thread]:
    specs = [
        ("task_camera", run_depthai_task_camera_worker),
        ("piper", run_piper_worker_basic),
        ("midas", run_midas_worker),
        ("teleop", run_teleop_worker),
    ]

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


def main_loop_basic(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    pedal: FootPedalTrigger,
    errors: "queue.Queue[WorkerError]",
) -> None:
    cv2 = import_cv2()
    pending_start_deadline: float | None = None
    recording_pedal_down = False

    while True:
        raise_next_worker_error(errors, recorder)
        if shared.shutdown.is_set():
            break

        task_bgr, _teleop_bgr = shared.display_frames()
        logitech_bgr = shared.latest_logitech_display_frame()
        if config.show_task_view and task_bgr is not None:
            cv2.imshow(config.task_view_window_name, task_bgr)
        if logitech_bgr is not None:
            cv2.imshow(config.logitech_window_name, logitech_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            shared.shutdown.set()
            break
        if key == ord("c"):
            shared.calibrate_neutral.set()

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
                pending_start_deadline = time.monotonic() + config.ready_seconds
                print(
                    f"Pedal pressed. Recording starts in {config.ready_seconds:.1f}s."
                )
        else:
            if pedal_pressed:
                print("Start cancelled by second pedal press during ready delay.")
                pending_start_deadline = None
            elif time.monotonic() >= pending_start_deadline:
                start_episode(config, recorder)
                pending_start_deadline = None


def run_collection_basic(config: CollectionConfig) -> int:
    config.dataset_root.mkdir(parents=True, exist_ok=True)
    raw_root(config.dataset_root).mkdir(parents=True, exist_ok=True)

    shared = SharedState()
    shared.set_teleop_enabled(config.teleop_enable_default)
    recorder = RecordingManager()
    errors: "queue.Queue[WorkerError]" = queue.Queue()

    pedal = FootPedalTrigger(
        config.foot_pedal_device,
        key_code=config.foot_pedal_key_code,
        debounce_s=config.foot_pedal_debounce_seconds,
    )
    pedal.start()
    print(f"Foot pedal input: {pedal.active_device}")
    print(
        "Drag-teach the Piper arm with its built-in teach button; hand "
        "tracking runs continuously via the Logitech camera."
    )
    print("Press c in a display window to calibrate teleop neutral. Press q to quit.")

    threads = start_workers_basic(config, shared, recorder, errors)
    deferred_error: WorkerError | None = None
    try:
        main_loop_basic(config, shared, recorder, pedal, errors)
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
