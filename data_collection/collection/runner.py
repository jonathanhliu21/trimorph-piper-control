from __future__ import annotations

import queue
import shutil
import threading
import time

from scripts.piper_api.foot_pedal import FootPedalTrigger

from .config import CollectionConfig
from .depthai_task_camera_worker import run_depthai_task_camera_worker
from .episode import EpisodeBuffer, RecordingManager, episode_paths, next_episode_index, raw_root
from .follower_hand_worker import run_follower_hand_worker
from .piper_worker import run_piper_worker
from .proprioception_worker import run_proprioception_worker
from .shared_state import SharedState
from .utils import WorkerError, import_cv2, worker_wrapper


_WORKER_INIT_TIMEOUT_S = 60.0
HOLD_DELETE_S = 2.0
_GREEN = "\033[32m"
_ORANGE = "\033[38;5;214m"
_RESET = "\033[0m"


def worker_specs(config: CollectionConfig):
    specs = [("piper", run_piper_worker), ("proprioception", run_proprioception_worker)]
    if config.task_camera_enabled:
        specs.insert(0, ("task_camera", run_depthai_task_camera_worker))
    if config.follower_hand_config is not None:
        specs.append(("follower", run_follower_hand_worker))
    return specs


def start_workers(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    errors: queue.Queue[WorkerError],
) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    try:
        for name, target in worker_specs(config):
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
            if not init_done.wait(timeout=_WORKER_INIT_TIMEOUT_S):
                raise TimeoutError(
                    f"{name} worker did not initialize within {_WORKER_INIT_TIMEOUT_S}s"
                )
            raise_next_worker_error(errors, recorder)
    except Exception:
        shared.shutdown.set()
        for thread in threads:
            thread.join(timeout=3.0)
        raise
    return threads


def _rate_hz(rows: list[dict[str, object]]) -> float:
    if len(rows) < 2:
        return 0.0
    elapsed_s = (int(rows[-1]["monotonic_ns"]) - int(rows[0]["monotonic_ns"])) / 1e9
    return 0.0 if elapsed_s <= 0 else (len(rows) - 1) / elapsed_s


def _print_episode_summary(session: EpisodeBuffer) -> None:
    duration_s = ((session.stop_ns or time.monotonic_ns()) - session.start_ns) / 1e9
    print(f"Saved episode {session.paths.episode_index:06d}: duration {duration_s:.3f}s")
    streams = [
        ("TRIMORPH proprioception", session.proprioception_rows, "samples"),
        ("Piper", session.piper_rows, "samples"),
    ]
    if session.config.task_camera_enabled:
        streams.insert(0, ("OAK camera", session.camera_rows, "frames"))
    if session.config.follower_hand_config is not None:
        streams.insert(-1, ("Follower MIDAS", session.follower_rows, "samples"))
    for name, rows, unit in streams:
        print(f"  {name}: {len(rows)} {unit}, active rate {_rate_hz(rows):.2f} Hz")


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
    session.flush()
    _print_episode_summary(session)
    if color:
        print(f"{color}Episode {session.paths.episode_index:06d} saved.{_RESET}")


def delete_and_discard_episode(recorder: RecordingManager, *, timeout_s: float = 5.0) -> None:
    session = recorder.stop("foot_pedal_delete")
    if session is None:
        return
    session.video_done.wait(timeout=timeout_s)
    shutil.rmtree(session.paths.episode_dir, ignore_errors=True)
    print(f"{_ORANGE}Episode {session.paths.episode_index:06d} deleted.{_RESET}")


def start_episode(config: CollectionConfig, recorder: RecordingManager) -> None:
    index = next_episode_index(raw_root(config.dataset_root))
    paths = episode_paths(config.dataset_root, index)
    paths.episode_dir.mkdir(parents=True, exist_ok=False)
    recorder.start(EpisodeBuffer(paths, config))
    print(f"Recording episode {index:06d}. Press pedal to save; hold {HOLD_DELETE_S:.0f}s to delete.")


def raise_next_worker_error(
    errors: queue.Queue[WorkerError], recorder: RecordingManager
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
    cv2 = import_cv2() if config.task_camera_enabled else None
    pending_start_deadline: float | None = None
    recording_pedal_down = False
    while not shared.shutdown.is_set():
        raise_next_worker_error(errors, recorder)
        frame = shared.latest_task_frame()
        if cv2 is not None and config.show_task_view and frame is not None:
            cv2.imshow(config.task_view_window_name, frame)
        key = cv2.waitKey(1) & 0xFF if cv2 is not None else -1
        if key in (ord("q"), 27):
            shared.shutdown.set()
            break

        pedal_pressed = bool(pedal.poll())
        if recorder.is_recording():
            hold_s = pedal.current_hold_s()
            if pedal_pressed and not recording_pedal_down:
                recording_pedal_down = True
                print(f"Pedal held. Release to save; hold {HOLD_DELETE_S:.0f}s to delete.")
            if recording_pedal_down:
                if hold_s is not None and hold_s >= HOLD_DELETE_S:
                    recording_pedal_down = False
                    delete_and_discard_episode(recorder)
                elif hold_s is None:
                    recording_pedal_down = False
                    stop_and_flush_episode(recorder, "foot_pedal", color=_GREEN)
            if cv2 is None:
                shared.shutdown.wait(0.01)
            continue

        if pending_start_deadline is None:
            if pedal_pressed:
                pending_start_deadline = time.monotonic() + config.ready_seconds
                print(f"Pedal pressed. Recording starts in {config.ready_seconds:.1f}s.")
        elif pedal_pressed:
            print("Start cancelled by second pedal press during ready delay.")
            pending_start_deadline = None
        elif time.monotonic() >= pending_start_deadline:
            start_episode(config, recorder)
            pending_start_deadline = None
        if cv2 is None:
            shared.shutdown.wait(0.01)


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
    quit_hint = "Press q to quit." if config.task_camera_enabled else "Press Ctrl-C to quit."
    print(f"Piper is passive: use its built-in drag-teach control. {quit_hint}")
    threads: list[threading.Thread] = []
    try:
        threads = start_workers(config, shared, recorder, errors)
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
        if config.task_camera_enabled:
            try:
                import_cv2().destroyAllWindows()
            except Exception:
                pass
    raise_next_worker_error(errors, recorder)
    return 0
