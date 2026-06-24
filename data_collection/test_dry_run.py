#!/usr/bin/env python3
"""Dry-run RGB-D wrist delta teleop without sending Piper motion commands."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from delta_teleop_controller import DeltaTeleopController, load_delta_teleop_config
from wrist_pose_estimator import estimate_wrist_pose_cam, matrix_to_euler_xyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print delta teleop target poses without commanding Piper.",
    )
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--cam-serial", default=None)
    parser.add_argument(
        "--ref-eef",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Use this Piper reference EEF pose instead of reading hardware.",
    )
    parser.add_argument(
        "--no-piper",
        action="store_true",
        help="Do not connect to Piper; requires --ref-eef.",
    )
    parser.add_argument("--piper-channel", default="can0")
    parser.add_argument("--piper-firmware", default="default")
    parser.add_argument("--piper-installation-pos", default="right")
    parser.add_argument("--print-hz", type=float, default=5.0)
    return parser.parse_args()


def _connect_piper(args: argparse.Namespace):
    if args.no_piper:
        if args.ref_eef is None:
            raise ValueError("--no-piper requires --ref-eef")
        return None
    if args.ref_eef is not None:
        return None

    from scripts.piper_api import PiperAPI

    api = PiperAPI(
        firmware=args.piper_firmware,
        channel=args.piper_channel,
        installation_pos=args.piper_installation_pos,
    )
    api.connect()
    print("Piper connected for read-only reference pose:", api.is_ok(), flush=True)
    return api


def _read_reference_eef(api, ref_eef: list[float] | None) -> list[float] | None:
    if ref_eef is not None:
        return list(ref_eef)
    if api is None:
        return None
    return api.get_eef_pose()


def main() -> int:
    args = parse_args()
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

    config = load_delta_teleop_config(args.config)
    controller = DeltaTeleopController(config)
    api = _connect_piper(args)

    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    if args.cam_serial:
        rs_config.enable_device(args.cam_serial)
    rs_config.enable_stream(rs.stream.color, args.cam_width, args.cam_height, rs.format.rgb8, args.cam_fps)
    rs_config.enable_stream(rs.stream.depth, args.cam_width, args.cam_height, rs.format.z16, args.cam_fps)
    align = rs.align(rs.stream.color)
    rs_pipeline.start(rs_config)
    print(
        f"RealSense started ({args.cam_width}x{args.cam_height}@{args.cam_fps} color+depth).",
        flush=True,
    )

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
    print(
        "Hold wrist still for calibration:",
        f"{config.calibration_frames} valid frames.",
        "No robot commands will be sent.",
        flush=True,
    )

    last_print = 0.0
    print_period = 1.0 / max(args.print_hz, 0.1)
    try:
        while True:
            frames = align.process(rs_pipeline.wait_for_frames(2000))
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
            wrist_pose = None
            if hand_frame is not None:
                teleop_frame = teleop.process_landmark_frame(hand_frame)
                detector.draw_landmarks(display, hand_frame)
                wrist_pose = estimate_wrist_pose_cam(
                    cv2,
                    hand_frame,
                    getattr(teleop_frame, "landmarks", None),
                    depth_m,
                    intrinsics,
                    width=display.shape[1],
                    height=display.shape[0],
                )

            ref_eef = _read_reference_eef(api, args.ref_eef)
            target_eef = controller.update(wrist_pose, ref_eef)
            status = controller.status()
            now = time.time()
            if now - last_print >= print_period:
                last_print = now
                if not status.calibrated:
                    print(
                        f"calibrating {status.calibration_count}/{status.calibration_frames}",
                        "tracking=ok" if wrist_pose is not None else "tracking=lost",
                        flush=True,
                    )
                elif target_eef is not None:
                    wrist_t = wrist_pose[:3, 3] if wrist_pose is not None else None
                    if wrist_pose is not None:
                        wr, wp, wy = matrix_to_euler_xyz(wrist_pose[:3, :3])
                        wrist_msg = (
                            f"wrist_cam_t=[{wrist_t[0]:+.3f} {wrist_t[1]:+.3f} {wrist_t[2]:+.3f}] "
                            f"rpy=[{wr:+.3f} {wp:+.3f} {wy:+.3f}]"
                        )
                    else:
                        wrist_msg = "wrist_cam=lost"
                    print(
                        "target_eef:",
                        f"[{target_eef[0]:+.3f} {target_eef[1]:+.3f} {target_eef[2]:+.3f} "
                        f"{target_eef[3]:+.3f} {target_eef[4]:+.3f} {target_eef[5]:+.3f}]",
                        "frozen" if status.frozen else "live",
                        wrist_msg,
                        flush=True,
                    )

            cv2.imshow("wrist_delta_teleop_dry_run", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        try:
            teleop.close()
        except Exception:
            pass
        rs_pipeline.stop()
        if api is not None:
            api.disconnect()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
