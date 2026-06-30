from __future__ import annotations

import time

from .config import CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .realsense_worker import flip_frame
from .shared_state import SharedState
from .utils import import_cv2

_WARMUP_ATTEMPTS = 60


def run_opencv_task_camera_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    cv2 = import_cv2()

    device = config.task_camera_device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise OSError(f"Could not open task camera: {config.task_camera_device!r}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    cap.set(cv2.CAP_PROP_FPS, config.camera_fps)

    actual_w = actual_h = None
    for _ in range(_WARMUP_ATTEMPTS):
        ok, frame = cap.read()
        if ok and frame is not None:
            actual_h, actual_w = frame.shape[:2]
            break
        time.sleep(0.05)

    if actual_w is None or actual_h is None:
        cap.release()
        raise RuntimeError(
            f"Task camera produced no frames during warmup "
            f"(device={config.task_camera_device!r})"
        )

    print(
        f"Task camera (opencv): {actual_w}x{actual_h} @{config.camera_fps}fps "
        f"device={config.task_camera_device!r}"
    )

    init_done.set()

    writer = None
    writer_session: EpisodeBuffer | None = None
    frame_idx = 0
    fourcc = cv2.VideoWriter_fourcc(*config.video_codec)

    try:
        while not shared.shutdown.is_set():
            ok, bgr = cap.read()
            if not ok or bgr is None:
                time.sleep(0.005)
                continue

            sample_ns = time.monotonic_ns()
            wall_time_s = time.time()

            bgr = flip_frame(cv2, bgr, config.camera_flip)
            shared.set_task_frame(bgr)

            session = recorder.current()
            if session is not writer_session:
                if writer is not None:
                    writer.release()
                    if writer_session is not None:
                        writer_session.video_done.set()
                writer = None
                writer_session = session
                frame_idx = 0
                if session is not None:
                    writer = cv2.VideoWriter(
                        str(session.paths.camera_path),
                        fourcc,
                        float(config.camera_fps),
                        (actual_w, actual_h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not open task camera video writer: "
                            f"{session.paths.camera_path}"
                        )

            if writer is not None and writer_session is not None:
                writer.write(bgr)
                writer_session.append_camera(
                    {
                        "frame_idx": frame_idx,
                        "monotonic_ns": sample_ns,
                        "wall_time_s": f"{wall_time_s:.6f}",
                        "elapsed_s": f"{writer_session.elapsed_s(sample_ns):.9f}",
                    }
                )
                frame_idx += 1
    finally:
        if writer is not None:
            writer.release()
            if writer_session is not None:
                writer_session.video_done.set()
        cap.release()
