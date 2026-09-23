#!/usr/bin/env python3
"""Collect raw OAK-1 task camera, RealSense, Piper, Midas teleop, and tactile episodes."""

from __future__ import annotations

import argparse
import glob
import os
import sys

from collection.config import (
    apply_device_overrides,
    build_config,
    validate_config,
)
from collection.realsense_worker import format_realsense_devices
from collection.runner import run_collection
from scripts.piper_api.foot_pedal import format_input_devices, list_input_devices

DATASET_ROOT = "data_midas7"  # e.g. "data_midas1" — overrides config.py; overridden by --dataset-root flag

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw Midas/Piper demonstrations. Most settings live in "
            "data_collection/collection/config.py; these flags are only for "
            "device selection and quick inspection."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Output folder for collected data. Overrides DATASET_ROOT in config.py.",
    )
    parser.add_argument(
        "--realsense-serial",
        "--realsense-id",
        dest="realsense_serial",
        default=None,
        help=(
            "RealSense serial number. Overrides REALSENSE_SERIAL. The RealSense "
            "only supplies depth for teleop wrist-pose estimation now — the task "
            "camera is always the OAK-1 (depthai)."
        ),
    )
    parser.add_argument(
        "--teleop-camera",
        default=None,
        help=(
            "OpenCV teleop camera index, video path, or /dev/v4l/by-id path. "
            "Overrides TELEOP_CAMERA."
        ),
    )
    parser.add_argument(
        "--foot-pedal-device",
        default=None,
        help="Foot pedal event device. Overrides FOOT_PEDAL_DEVICE.",
    )
    parser.add_argument(
        "--curobo",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable cuRobo motion generation. Use --no-curobo to fall back to "
            "firmware IK. Defaults to CUROBO_ENABLED from config.py."
        ),
    )
    parser.add_argument(
        "--aux-camera",
        dest="aux_camera_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable the auxiliary OAK-1 camera (Luxonis depthai; a separate "
            "physical device from the task and teleop cameras, auto-detected over "
            "USB). Defaults to AUX_CAMERA_ENABLED from config.py. Requires the "
            "'depthai' package. Saved as wrist_camera.mp4 for dataset compatibility."
        ),
    )
    parser.add_argument(
        "--list-realsense",
        action="store_true",
        help="List RealSense devices/profiles and exit.",
    )
    parser.add_argument(
        "--list-input-devices",
        action="store_true",
        help="List /dev/input event devices and exit.",
    )
    parser.add_argument(
        "--list-video-devices",
        action="store_true",
        help="List Linux video devices and stable /dev/v4l/by-id links.",
    )
    return parser.parse_args()


def format_video_devices() -> str:
    link_map = {}
    for link in glob.glob("/dev/v4l/by-id/*") + glob.glob("/dev/v4l/by-path/*"):
        target = os.path.realpath(link)
        if os.path.basename(target).startswith("video"):
            link_map.setdefault(target, []).append(link)

    paths = set(glob.glob("/dev/video*"))
    paths.update(link_map)
    if not paths:
        return "No /dev/video* devices found."

    lines = []
    for path in sorted(paths):
        real = os.path.realpath(path)
        lines.append(path)
        for link in sorted(link_map.get(real, [])):
            lines.append(f"  {link}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.list_input_devices:
        print(format_input_devices(list_input_devices()))
        return 0
    if args.list_video_devices:
        print(format_video_devices())
        return 0
    if args.list_realsense:
        print(format_realsense_devices())
        return 0

    config = apply_device_overrides(
        build_config(),
        dataset_root=args.dataset_root or DATASET_ROOT,
        realsense_serial=args.realsense_serial,
        teleop_camera=args.teleop_camera,
        foot_pedal_device=args.foot_pedal_device,
        curobo_enabled=args.curobo,
        aux_camera_enabled=args.aux_camera_enabled,
    )
    validate_config(config)
    return run_collection(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
