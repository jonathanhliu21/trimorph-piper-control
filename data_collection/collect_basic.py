#!/usr/bin/env python3
"""Lightweight data collection: OAK-1 task camera + Logitech hand-tracking +
Piper native drag-teach + Midas hand teleop. No MIT/backdrive/cuRobo,
RealSense, or tactile — see collect_raw_dataset.py for the full pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from collection.config import apply_device_overrides, build_config, validate_config
from collection.runner_basic import run_collection_basic
from scripts.piper_api.foot_pedal import format_input_devices, list_input_devices

DATASET_ROOT = "data_midas7                           "  # e.g. "data_midas7" — overrides config.py; overridden by --dataset-root flag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Midas/Piper drag-teach demonstrations with a minimal set "
            "of workers (task camera, Logitech hand tracking, Piper passive "
            "logging, Midas hand teleop). Most settings live in "
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
        "--teleop-camera",
        default=None,
        help=(
            "OpenCV hand-tracking camera index, video path, or /dev/v4l/by-id "
            "path (Logitech cam). Overrides TELEOP_CAMERA."
        ),
    )
    parser.add_argument(
        "--foot-pedal-device",
        default=None,
        help="Foot pedal event device. Overrides FOOT_PEDAL_DEVICE.",
    )
    parser.add_argument(
        "--list-input-devices",
        action="store_true",
        help="List /dev/input event devices and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_input_devices:
        print(format_input_devices(list_input_devices()))
        return 0

    config = apply_device_overrides(
        build_config(),
        dataset_root=args.dataset_root or DATASET_ROOT,
        teleop_camera=args.teleop_camera,
        foot_pedal_device=args.foot_pedal_device,
    )
    config = replace(
        config,
        teleop_source="opencv",
        record_teleop_camera=False,
        enable_tactile=False,
        require_tactile=False,
        aux_camera_enabled=False,
        curobo_enabled=False,
        teleop_enable_default=True,
    )
    validate_config(config)
    return run_collection_basic(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
