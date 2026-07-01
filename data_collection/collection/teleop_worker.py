from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from wrist_pose_estimator import estimate_wrist_pose_cam

from .config import CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .shared_state import SharedState
from .utils import import_cv2


def run_teleop_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    cv2 = import_cv2()
    from midas_hand_retargeter import MidasHandRetargeter
    from midas_hand_retargeter.adaptor import PIP_DIP_LOOKUP_MODE
    from midas_hand_teleop.detector import MediaPipeHandDetector
    from midas_hand_teleop.pipeline import MidasTeleopPipeline

    import midas_hand_teleop as _teleop_pkg
    pinch_calibration = None
    try:
        from midas_hand_teleop.pinch_calibration import (
            PinchCalibration,
            extract_pinch_features,
            load_pinch_calibration,
        )
        _pinch_path = (
            Path(_teleop_pkg.__file__).parent.parent / "config" / "pinch_calibration.json"
        )
        if _pinch_path.exists():
            pinch_calibration = load_pinch_calibration(_pinch_path)
            print(f"Loaded pinch calibration from {_pinch_path}")
        else:
            print("No pinch calibration found; running without pinch correction.")
    except ImportError:
        print("Pinch calibration not available in this version of midas_hand_teleop.")

    cap = None
    if config.teleop_source == "opencv":
        camera = (
            int(config.teleop_camera)
            if str(config.teleop_camera).isdigit()
            else config.teleop_camera
        )
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise OSError(f"Could not open teleop camera/video: {config.teleop_camera}")

    # Logitech cap for runtime z-key switching (realsense mode only)
    logitech_cap = None
    if config.teleop_source == "realsense":
        logitech_device = (
            int(config.teleop_camera)
            if str(config.teleop_camera).isdigit()
            else config.teleop_camera
        )
        _lc = cv2.VideoCapture(logitech_device)
        if _lc.isOpened():
            logitech_cap = _lc
            print(f"Logitech finger camera opened: {config.teleop_camera} (press z to switch)")
        else:
            _lc.release()
            print(f"Warning: Could not open Logitech camera at {config.teleop_camera}; z-key switching unavailable.")

    detector = MediaPipeHandDetector(
        hand_type=config.teleop_hand_type,
        input_hand_type=config.teleop_input_hand,
        selfie=config.teleop_selfie,
        mirror_input=config.teleop_mirror_input,
        mirror_axis=config.teleop_mirror_axis,
        lock_input_hand=config.teleop_lock_input_hand,
        model_path=config.teleop_hand_landmarker_model,
    )
    print("MediaPipe hand detector initialized.")
    retargeter = MidasHandRetargeter.create(
        scaling_factor=config.teleop_scaling_factor,
        coupling_mode=PIP_DIP_LOOKUP_MODE,
    )
    print("Midas retargeter initialized.")
    pipeline = MidasTeleopPipeline(detector=detector, retargeter=retargeter)
    init_done.set()

    writer = None
    writer_session: EpisodeBuffer | None = None
    frame_idx = 0
    fourcc = cv2.VideoWriter_fourcc(*config.video_codec)
    last_teleop_frame = None
    try:
        while not shared.shutdown.is_set():
            use_logitech = shared.logitech_finger_mode() and logitech_cap is not None

            depth_m = None
            intrinsics = None
            logitech_raw: "np.ndarray | None" = None

            if config.teleop_source == "opencv":
                assert cap is not None
                ok, bgr = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
            elif use_logitech:
                ok, bgr = logitech_cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
            else:
                bgr, depth_m, intrinsics = shared.latest_realsense_rgbd()
                if bgr is None:
                    time.sleep(0.005)
                    continue
                # Read Logitech frame for display even when not the active input.
                # The cap buffer is always draining so this read is nearly instant.
                if logitech_cap is not None:
                    ok_l, _lbgr = logitech_cap.read()
                    if ok_l:
                        logitech_raw = _lbgr
            sample_ns = time.monotonic_ns()
            wall_time_s = time.time()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            hand_frame = detector.detect_rgb(rgb)
            display = bgr.copy()

            if hand_frame is not None:
                teleop_frame = pipeline.process_landmark_frame(hand_frame)
                if pinch_calibration is not None:
                    pinch_features = extract_pinch_features(teleop_frame.landmarks)
                    teleop_frame = replace(
                        teleop_frame,
                        retargeting=pinch_calibration.apply(teleop_frame.retargeting, pinch_features),
                    )
                last_teleop_frame = teleop_frame
                teacher_active, _ = shared.teacher_mode()
                if teacher_active or shared.teleop_enabled():
                    shared.set_midas_target(teleop_frame.hardware_motor_positions)
                # Wrist pose estimate in the RealSense camera optical frame (4x4).
                if depth_m is not None and intrinsics is not None:
                    pose = estimate_wrist_pose_cam(
                        cv2,
                        hand_frame,
                        getattr(teleop_frame, "landmarks", None),
                        depth_m,
                        intrinsics,
                        width=display.shape[1],
                        height=display.shape[0],
                    )
                    shared.set_wrist_pose_cam(pose)
                detector.draw_landmarks(display, hand_frame)
            elif pinch_calibration is not None:
                pinch_calibration.update_confidence(None)
                shared.set_wrist_pose_cam(None)

            if shared.calibrate_neutral.is_set():
                shared.calibrate_neutral.clear()
                if last_teleop_frame is None:
                    print("No teleop frame available for neutral calibration.")
                else:
                    offsets = pipeline.calibrate_neutral_from_last_frame()
                    compact = {name: round(value, 3) for name, value in offsets.items()}
                    print(f"Captured retargeter neutral offsets: {compact}")

            shared.set_teleop_frame(display)

            # Keep the Logitech window updated regardless of which source is active.
            if use_logitech:
                shared.set_logitech_display_frame(display)  # with landmarks
            elif logitech_raw is not None:
                shared.set_logitech_display_frame(logitech_raw)  # raw feed

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
        if cap is not None:
            cap.release()
        if logitech_cap is not None:
            logitech_cap.release()
