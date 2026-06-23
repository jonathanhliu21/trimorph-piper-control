from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .shared_state import SharedState
from .utils import import_cv2


def _landmark_xy_pixels(hand_frame, width: int, height: int) -> np.ndarray | None:
    # `HandLandmarkFrame.landmarks` are wrist-centered 3D landmarks (MANO-ish),
    # not image pixels. For pixel positions, use `image_landmarks`.
    landmarks = getattr(hand_frame, "image_landmarks", None)
    if landmarks is None:
        return None
    try:
        if hasattr(landmarks, "landmark"):
            landmarks = getattr(landmarks, "landmark")
        raw = []
        for lm in landmarks:
            if hasattr(lm, "x") and hasattr(lm, "y"):
                x = float(lm.x)
                y = float(lm.y)
            else:
                x = float(lm[0])
                y = float(lm[1])
            raw.append([x, y])
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        if not np.isfinite(arr).all():
            return None

        # Auto-detect whether landmarks are normalized (0..1) or already in pixels.
        x = arr[:, 0]
        y = arr[:, 1]
        normalized = (
            np.all((x >= -0.25) & (x <= 1.25))
            and np.all((y >= -0.25) & (y <= 1.25))
        )
        if normalized:
            arr = np.stack([x * width, y * height], axis=1)
        else:
            # Assume pixel coordinates.
            arr = np.stack([x, y], axis=1)
        return arr
    except Exception:
        return None


def _as_xyz_array(points) -> np.ndarray | None:
    if points is None:
        return None
    try:
        rows = []
        for p in points:
            if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
                x = float(p.x)
                y = float(p.y)
                z = float(p.z)
            else:
                x = float(p[0])
                y = float(p[1])
                z = float(p[2])
            rows.append([x, y, z])
        arr = np.asarray(rows, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None
        if not np.isfinite(arr).all():
            return None
        return arr
    except Exception:
        return None


def _camera_points_from_depth(
    pixels_xy: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])
    h, w = depth_m.shape[:2]
    pts = []
    valid = []
    for i, (u, v) in enumerate(pixels_xy):
        ui = int(round(float(u)))
        vi = int(round(float(v)))
        if ui < 0 or vi < 0 or ui >= w or vi >= h:
            valid.append(False)
            pts.append([math.nan, math.nan, math.nan])
            continue
        z = float(depth_m[vi, ui])
        if not math.isfinite(z) or z <= 0:
            valid.append(False)
            pts.append([math.nan, math.nan, math.nan])
            continue
        x = (float(ui) - ppx) / fx * z
        y = (float(vi) - ppy) / fy * z
        pts.append([x, y, z])
        valid.append(True)
    return np.asarray(pts, dtype=np.float64), np.asarray(valid, dtype=bool)


def _rigid_transform(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < 4:
        return None
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    aa = a - a_mean
    bb = b - b_mean
    h = aa.T @ bb
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = b_mean - r @ a_mean
    return r, t


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
            depth_m = None
            intrinsics = None
            if config.teleop_source == "opencv":
                assert cap is not None
                ok, bgr = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
            else:
                bgr, depth_m, intrinsics = shared.latest_realsense_rgbd()
                if bgr is None:
                    time.sleep(0.005)
                    continue
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
                if shared.teleop_enabled():
                    shared.set_midas_target(teleop_frame.hardware_motor_positions)
                # Wrist pose estimate in the RealSense camera frame (4x4).
                if depth_m is not None and intrinsics is not None:
                    pixels_xy = _landmark_xy_pixels(hand_frame, display.shape[1], display.shape[0])
                    wrist_points = _as_xyz_array(getattr(teleop_frame, "landmarks", None))
                    if pixels_xy is not None and wrist_points is not None and len(pixels_xy) == len(wrist_points):
                        cam_points, valid = _camera_points_from_depth(pixels_xy, depth_m, intrinsics)
                        ok = valid & np.all(np.isfinite(cam_points), axis=1) & np.all(np.isfinite(wrist_points), axis=1)
                        if int(np.count_nonzero(ok)) >= 6:
                            fx = float(intrinsics["fx"])
                            fy = float(intrinsics["fy"])
                            ppx = float(intrinsics["ppx"])
                            ppy = float(intrinsics["ppy"])
                            camera_matrix = np.asarray(
                                [[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]],
                                dtype=np.float64,
                            )
                            dist = np.zeros((4, 1), dtype=np.float64)
                            obj = wrist_points[ok].astype(np.float64, copy=False)
                            img = pixels_xy[ok].astype(np.float64, copy=False)
                            success = False
                            try:
                                success, rvec, tvec, _inliers = cv2.solvePnPRansac(
                                    obj,
                                    img,
                                    camera_matrix,
                                    distCoeffs=dist,
                                    flags=cv2.SOLVEPNP_EPNP,
                                    reprojectionError=6.0,
                                    confidence=0.99,
                                    iterationsCount=100,
                                )
                            except Exception:
                                success = False
                            if success:
                                rmat, _ = cv2.Rodrigues(rvec)
                                pose = np.eye(4, dtype=np.float64)
                                pose[:3, :3] = np.asarray(rmat, dtype=np.float64)
                                pose[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
                                shared.set_wrist_pose_cam(pose)
                            else:
                                r_t = _rigid_transform(obj, cam_points[ok])
                                if r_t is not None:
                                    r, t = r_t
                                    pose = np.eye(4, dtype=np.float64)
                                    pose[:3, :3] = r
                                    pose[:3, 3] = t
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
