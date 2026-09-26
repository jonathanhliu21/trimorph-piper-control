#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from collection.config import apply_overrides, build_config, validate_config
from collection.runner import run_collection
from scripts.piper_api.foot_pedal import format_input_devices, list_input_devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect synchronized OAK, PiperX, and hand proprioception, with optional follower."
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--hand-config", "--leader-config", dest="hand_config")
    parser.add_argument("--hand-port", "--leader-port", dest="hand_port")
    parser.add_argument("--follower-config")
    parser.add_argument("--follower-port")
    parser.add_argument("--foot-pedal-device")
    parser.add_argument("--list-input-devices", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_input_devices:
        print(format_input_devices(list_input_devices()))
        return 0
    config = apply_overrides(
        build_config(),
        dataset_root=args.dataset_root,
        hand_config=args.hand_config,
        hand_port=args.hand_port,
        follower_config=args.follower_config,
        follower_port=args.follower_port,
        foot_pedal_device=args.foot_pedal_device,
    )
    validate_config(config)
    return run_collection(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
