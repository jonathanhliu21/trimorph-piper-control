#!/usr/bin/env python3
"""
Quick manual test: stream RealSense RGB-D, detect hand landmarks, and estimate
the wrist pose in the RealSense camera frame using aligned depth + intrinsics.

This reuses the existing teleop stack (MediaPipe detector + MidasTeleopPipeline)
and the wrist pose logic added in `data_collection/collection/teleop_worker.py`.

Usage:
  python3 data_collection/test_realsense_wrist_pose.py

Controls:
  - q / Esc: quit
"""

from __future__ import annotations

import math
import time

import numpy as np


def _matrix_to_euler_xyz(r: np.ndarray) -> tuple[float, float, float]:
    # fixed-axis XYZ: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    pitch = math.asin(float(-r[2, 0]))
    cp = math.cos(pitch)
    if abs(cp) < 1e-9:
        roll = 0.0
        yaw = math.atan2(float(-r[0, 1]), float(r[1, 1]))
    else:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return roll, pitch, yaw


def _image_landmarks_to_pixels(image_landmarks, width: int, height: int) -> np.ndarray | None:
    if image_landmarks is None:
        return None
    try:
        if hasattr(image_landmarks, "landmark"):
            image_landmarks = getattr(image_landmarks, "landmark")
        pts = []
        for lm in image_landmarks:
            if hasattr(lm, "x") and hasattr(lm, "y"):
                x = float(lm.x)
                y = float(lm.y)
            else:
                x = float(lm[0])
                y = float(lm[1])
            pts.append([x * width, y * height])
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
            return None
        if not np.isfinite(arr).all():
            return None
        return arr
    except Exception:
        return None


def main() -> int:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing OpenCV Python bindings. Install with: python3 -m pip install opencv-python"
        ) from exc

    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Intel RealSense Python bindings. Install librealsense/pyrealsense2."
        ) from exc

    # Existing teleop stack.
    from midas_hand_retargeter import MidasHandRetargeter
    from midas_hand_retargeter.adaptor import PIP_DIP_LOOKUP_MODE
    from midas_hand_teleop.detector import MediaPipeHandDetector
    from midas_hand_teleop.pipeline import MidasTeleopPipeline

    # Reuse helpers from the collector’s teleop worker.
    from collection.teleop_worker import _rigid_transform

    print("Starting RealSense wrist pose test...", flush=True)

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    pipeline.start(rs_config)
    print("RealSense started (640x480@30 color+depth, aligned).", flush=True)

    detector = MediaPipeHandDetector(
        hand_type="Right",
        input_hand_type="auto",
        selfie=False,
        mirror_input=True,
        mirror_axis="y",
        lock_input_hand=False,
        model_path=None,
    )
    retargeter = MidasHandRetargeter.create(scaling_factor=1.15, coupling_mode=PIP_DIP_LOOKUP_MODE)
    teleop = MidasTeleopPipeline(detector=detector, retargeter=retargeter)
    print("MediaPipe detector + retargeter initialized.", flush=True)

    last_print = 0.0
    try:
        while True:
            frames = pipeline.wait_for_frames(2000)
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            rgb = np.asanyarray(color.get_data())
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            intr = color.profile.as_video_stream_profile().intrinsics
            intrinsics = {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "ppx": float(intr.ppx),
                "ppy": float(intr.ppy),
            }
            depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * float(depth.get_units())

            hand_frame = detector.detect_rgb(rgb)
            display = bgr.copy()
            pose = None
            pixels_xy = None

            if hand_frame is not None:
                teleop_frame = teleop.process_landmark_frame(hand_frame)
                detector.draw_landmarks(display, hand_frame)

                # 2D pixels from the detector frame.
                pixels_xy = _image_landmarks_to_pixels(
                    getattr(hand_frame, "image_landmarks", None),
                    display.shape[1],
                    display.shape[0],
                )
                # 3D keypoints in the local wrist frame from the existing pipeline output.
                wrist_points = np.asarray(getattr(teleop_frame, "landmarks", None), dtype=np.float64)
                if (
                    pixels_xy is not None
                    and wrist_points.ndim == 2
                    and wrist_points.shape[1] == 3
                    and len(pixels_xy) == len(wrist_points)
                ):
                    ok = np.isfinite(wrist_points).all(axis=1)
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
                        else:
                            # Fallback: form 3D camera points using depth then rigid-fit.
                            cam_pts = []
                            valid = []
                            h, w = depth_m.shape[:2]
                            for (u, v) in img:
                                ui = int(round(float(u)))
                                vi = int(round(float(v)))
                                if ui < 0 or vi < 0 or ui >= w or vi >= h:
                                    cam_pts.append([math.nan, math.nan, math.nan])
                                    valid.append(False)
                                    continue
                                z = float(depth_m[vi, ui])
                                if not math.isfinite(z) or z <= 0:
                                    cam_pts.append([math.nan, math.nan, math.nan])
                                    valid.append(False)
                                    continue
                                x = (float(ui) - ppx) / fx * z
                                y = (float(vi) - ppy) / fy * z
                                cam_pts.append([x, y, z])
                                valid.append(True)
                            cam_pts = np.asarray(cam_pts, dtype=np.float64)
                            valid = np.asarray(valid, dtype=bool) & np.isfinite(cam_pts).all(axis=1)
                            if int(np.count_nonzero(valid)) >= 6:
                                rt = _rigid_transform(obj[valid], cam_pts[valid])
                                if rt is not None:
                                    r, t = rt
                                    pose = np.eye(4, dtype=np.float64)
                                    pose[:3, :3] = r
                                    pose[:3, 3] = t

            # Print only when a pose is available.
            now = time.time()
            if pose is not None and now - last_print >= 0.2:
                last_print = now
                roll, pitch, yaw = _matrix_to_euler_xyz(pose[:3, :3])
                t = pose[:3, 3]
                print(
                    "wrist_pose_cam:",
                    f"t=[{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}]",
                    f"rpy=[{roll:+.3f} {pitch:+.3f} {yaw:+.3f}] rad",
                    flush=True,
                )

            # Always show video.
            cv2.imshow("realsense_wrist_pose_test", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
b    finally:
        try:
            teleop.close()
        except Exception:
            pass
        pipeline.stop()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

