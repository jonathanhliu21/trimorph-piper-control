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

import time

import numpy as np

from wrist_pose_estimator import estimate_wrist_pose_cam, matrix_to_euler_xyz


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

            if hand_frame is not None:
                teleop_frame = teleop.process_landmark_frame(hand_frame)
                detector.draw_landmarks(display, hand_frame)
                pose = estimate_wrist_pose_cam(
                    cv2,
                    hand_frame,
                    getattr(teleop_frame, "landmarks", None),
                    depth_m,
                    intrinsics,
                    width=display.shape[1],
                    height=display.shape[0],
                )

            # Print only when a pose is available.
            now = time.time()
            if pose is not None and now - last_print >= 0.2:
                last_print = now
                roll, pitch, yaw = matrix_to_euler_xyz(pose[:3, :3])
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
    finally:
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

