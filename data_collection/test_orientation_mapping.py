#!/usr/bin/env python3
"""
Offline sanity check for the r_cam2robot ORIENTATION mapping used by
DeltaTeleopController when orientation_enabled=True.

Never touches the Piper arm -- streams RealSense + MediaPipe wrist pose only
(same stack as test_realsense_wrist_pose.py) and prints the same
delta_rot_cam -> delta_rot_robot conjugation from
delta_teleop_controller.py's _raw_target_pose(), so you can rotate your wrist
about one axis at a time and read off whether the mapped robot-frame
rotation lands on a sane axis/sign before re-enabling orientation_enabled
on real hardware.

Usage:
  python3 data_collection/test_orientation_mapping.py

Controls:
  - c: set current wrist orientation as the calibration reference (do this
       with your wrist held still, wrist pointing however feels neutral)
  - q / Esc: quit
"""

from __future__ import annotations

import time

import numpy as np

from delta_teleop_controller import load_delta_teleop_config
from wrist_pose_estimator import estimate_wrist_pose_cam, matrix_to_euler_xyz

RAD2DEG = 180.0 / np.pi


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

    from midas_hand_retargeter import MidasHandRetargeter
    from midas_hand_retargeter.adaptor import PIP_DIP_LOOKUP_MODE
    from midas_hand_teleop.detector import MediaPipeHandDetector
    from midas_hand_teleop.pipeline import MidasTeleopPipeline

    config = load_delta_teleop_config()
    r_cam2robot = config.r_cam2robot
    r_cam2robot_inv = np.linalg.inv(r_cam2robot)
    print(f"Loaded r_cam2robot from config.yaml:\n{r_cam2robot}", flush=True)

    print("Starting orientation mapping test (no robot connection)...", flush=True)

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
    print(
        "\nPress 'c' with wrist held still to set the reference orientation.\n"
        "Then rotate about ONE axis at a time (roll / pitch / yaw) and check\n"
        "which robot-frame axis lights up and whether the sign matches your\n"
        "expectation. Press q/Esc to quit.\n",
        flush=True,
    )

    ref_rot_cam: np.ndarray | None = None
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

            now = time.time()
            if pose is not None and ref_rot_cam is not None and now - last_print >= 0.2:
                last_print = now
                cur_rot_cam = pose[:3, :3]
                delta_rot_cam = ref_rot_cam.T @ cur_rot_cam
                delta_rot_robot = r_cam2robot @ delta_rot_cam @ r_cam2robot_inv

                cam_r, cam_p, cam_y = matrix_to_euler_xyz(delta_rot_cam)
                rob_r, rob_p, rob_y = matrix_to_euler_xyz(delta_rot_robot)
                print(
                    f"delta_cam(deg)  roll={cam_r * RAD2DEG:+6.1f} "
                    f"pitch={cam_p * RAD2DEG:+6.1f} yaw={cam_y * RAD2DEG:+6.1f}   "
                    f"|   delta_robot(deg)  roll={rob_r * RAD2DEG:+6.1f} "
                    f"pitch={rob_p * RAD2DEG:+6.1f} yaw={rob_y * RAD2DEG:+6.1f}",
                    flush=True,
                )

            cv2.putText(
                display,
                "c: set reference   q/Esc: quit",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("orientation_mapping_test", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c") and pose is not None:
                ref_rot_cam = pose[:3, :3].copy()
                print(">>> Reference orientation set.", flush=True)
    finally:
        try:
            teleop.close()
        except Exception:
            pass
        pipeline.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
