from __future__ import annotations

import time

from .config import CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .shared_state import SharedState
from .utils import import_cv2


def run_teleop_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
) -> None:
    cv2 = import_cv2()
    from midas_hand_retargeter import MidasHandRetargeter
    from midas_hand_retargeter.adaptor import PIP_DIP_LOOKUP_MODE
    from midas_hand_teleop.detector import MediaPipeHandDetector
    from midas_hand_teleop.pipeline import MidasTeleopPipeline

    camera = (
        int(config.teleop_camera)
        if str(config.teleop_camera).isdigit()
        else config.teleop_camera
    )
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise OSError(f"Could not open teleop camera/video: {config.teleop_camera}")

    detector = MediaPipeHandDetector(
        hand_type=config.teleop_hand_type,
        input_hand_type=config.teleop_input_hand,
        selfie=config.teleop_selfie,
        mirror_input=config.teleop_mirror_input,
        mirror_axis=config.teleop_mirror_axis,
        lock_input_hand=config.teleop_lock_input_hand,
        model_path=config.teleop_hand_landmarker_model,
    )
    retargeter = MidasHandRetargeter.create(
        scaling_factor=config.teleop_scaling_factor,
        coupling_mode=PIP_DIP_LOOKUP_MODE,
    )
    pipeline = MidasTeleopPipeline(detector=detector, retargeter=retargeter)

    writer = None
    writer_session: EpisodeBuffer | None = None
    frame_idx = 0
    fourcc = cv2.VideoWriter_fourcc(*config.video_codec)
    last_teleop_frame = None
    try:
        while not shared.shutdown.is_set():
            ok, bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            sample_ns = time.monotonic_ns()
            wall_time_s = time.time()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hand_frame = detector.detect_rgb(rgb)
            display = bgr.copy()

            if hand_frame is not None:
                teleop_frame = pipeline.process_landmark_frame(hand_frame)
                last_teleop_frame = teleop_frame
                teacher_active, _ = shared.teacher_mode()
                if teacher_active:
                    shared.set_midas_target(teleop_frame.hardware_motor_positions)
                detector.draw_landmarks(display, hand_frame)

            if shared.calibrate_neutral.is_set():
                shared.calibrate_neutral.clear()
                if last_teleop_frame is None:
                    print("No teleop frame available for neutral calibration.")
                else:
                    offsets = pipeline.calibrate_neutral_from_last_frame()
                    compact = {name: round(value, 3) for name, value in offsets.items()}
                    print(f"Captured retargeter neutral offsets: {compact}")

            shared.set_teleop_frame(display)

            session = recorder.current()
            if not config.record_teleop_camera:
                session = None
            if session is not writer_session:
                if writer is not None:
                    writer.release()
                    if writer_session is not None:
                        writer_session.teleop_video_done.set()
                writer = None
                writer_session = session
                frame_idx = 0
                if session is not None:
                    height, width = display.shape[:2]
                    writer = cv2.VideoWriter(
                        str(session.paths.teleop_camera_path),
                        fourcc,
                        float(config.teleop_camera_fps),
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open {session.paths.teleop_camera_path}")

            if writer is not None and writer_session is not None:
                writer.write(display)
                writer_session.append_teleop_camera(
                    {
                        "frame_idx": frame_idx,
                        "monotonic_ns": sample_ns,
                        "wall_time_s": f"{wall_time_s:.6f}",
                        "elapsed_s": f"{writer_session.elapsed_s(sample_ns):.9f}",
                        "hand_detected": hand_frame is not None,
                    }
                )
                frame_idx += 1
    finally:
        if writer is not None:
            writer.release()
            if writer_session is not None:
                writer_session.teleop_video_done.set()
        pipeline.close()
        cap.release()
