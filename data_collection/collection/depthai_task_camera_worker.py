from __future__ import annotations

import time

from .config import CollectionConfig
from .depthai_utils import import_depthai, require_depthai_device, warmup_color_queue
from .episode import EpisodeBuffer, RecordingManager
from .shared_state import SharedState
from .utils import import_cv2

_WARMUP_ATTEMPTS = 60


def flip_frame(cv2, frame, mode: str):
    if mode == "horizontal":
        return cv2.flip(frame, 1)
    if mode == "vertical":
        return cv2.flip(frame, 0)
    if mode == "both":
        return cv2.flip(frame, -1)
    return frame


def run_depthai_task_camera_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    cv2 = import_cv2()
    dai = import_depthai()
    require_depthai_device(dai)

    writer = None
    writer_session: EpisodeBuffer | None = None
    frame_idx = 0
    fourcc = cv2.VideoWriter_fourcc(*config.video_codec)

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(
            (config.task_camera_width, config.task_camera_height), dai.ImgFrame.Type.NV12
        )
        queue = cam_out.createOutputQueue()
        pipeline.start()

        size = warmup_color_queue(queue, _WARMUP_ATTEMPTS)
        if size is None:
            raise RuntimeError("Task camera (OAK-1) produced no frames during warmup")
        actual_w, actual_h = size

        print(f"Task camera (depthai/OAK-1): {actual_w}x{actual_h} @{config.task_camera_fps}fps")
        init_done.set()

        try:
            while not shared.shutdown.is_set() and pipeline.isRunning():
                bgr = queue.get().getCvFrame()
                if bgr is None:
                    time.sleep(0.005)
                    continue

                sample_ns = time.monotonic_ns()
                wall_time_s = time.time()

                bgr = flip_frame(cv2, bgr, config.task_camera_flip)
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
                            float(config.task_camera_fps),
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
