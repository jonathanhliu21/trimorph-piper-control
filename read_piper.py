#!/usr/bin/env python3
"""Read Piper joint angles and end-effector pose without commanding motion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Sequence

from scripts.piper_api.piper_api import PiperAPI


JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))
EEF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read all Piper joint angles and the EEF position/RPY pose."
    )
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can0")
    parser.add_argument(
        "--firmware", choices=("default", "v183", "v188"), default="v183"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one reading and exit (the default is to stream until Ctrl-C).",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=10.0,
        help="Streaming output rate (default: 10).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3.0,
        help="How long to wait for a complete reading (default: 3).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object per reading.",
    )
    return parser.parse_args(argv)


def read_sample(api: PiperAPI, timeout_s: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while True:
        joints = api.get_joint_state()
        eef = api.get_eef_state()
        if joints is not None and eef is not None:
            return {
                "timestamp_s": max(joints.timestamp, eef.timestamp),
                "joints_rad": dict(zip(JOINT_NAMES, joints.joints)),
                "eef_position_m": dict(zip(EEF_NAMES[:3], eef.pose[:3])),
                "eef_rpy_rad": dict(zip(EEF_NAMES[3:], eef.pose[3:])),
            }
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for joint-angle and EEF-pose feedback"
            )
        time.sleep(0.01)


def format_sample(sample: dict[str, object]) -> str:
    joints = sample["joints_rad"]
    position = sample["eef_position_m"]
    rpy = sample["eef_rpy_rad"]
    assert isinstance(joints, dict)
    assert isinstance(position, dict)
    assert isinstance(rpy, dict)
    return "\n".join(
        (
            "joints [rad]: "
            + "  ".join(f"{name}={value:+.6f}" for name, value in joints.items()),
            "eef pos [m]:  "
            + "  ".join(f"{name}={value:+.6f}" for name, value in position.items()),
            "eef rpy [rad]: "
            + "  ".join(f"{name}={value:+.6f}" for name, value in rpy.items()),
        )
    )


def run(args: argparse.Namespace) -> int:
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be greater than 0")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be greater than 0")

    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
        installation_pos=None,
        payload=None,
    )
    try:
        api.connect()
        period_s = 1.0 / args.rate_hz
        while True:
            sample = read_sample(api, args.timeout_s)
            if args.json:
                print(json.dumps(sample), flush=True)
            else:
                print(format_sample(sample), flush=True)
            if args.once:
                return 0
            time.sleep(period_s)
    finally:
        api.disconnect()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except (KeyError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
