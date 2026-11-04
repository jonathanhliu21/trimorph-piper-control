#!/usr/bin/env python3
"""Move Piper in joint space to a configuration for an in-person check, then hold or return.

The default target is the median start of data_trimorph_hand_mocap_quest_0 (flange about
(144, -202, 270) mm in the Piper base); --home picks another named start from
PIPER_START_JOINTS_DEG in collection/config.py. Kinematics-only checked: watch for the mount and desk.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from data_collection.collection.config import (  # noqa: E402
    DEFAULT_PIPER_CHANNEL,
    DEFAULT_PIPER_FIRMWARE,
    DEFAULT_PIPER_INSTALLATION_POS,
    DEFAULT_PIPER_INTERFACE,
    DEFAULT_PIPER_PAYLOAD,
    PIPER_START_JOINTS_DEG,
)
from scripts.piper_api.piper_api import PiperAPI  # noqa: E402
from scripts.piper_api.replay_trajectory import read_joint_angles  # noqa: E402
from scripts.teleop.quest_teleop import (  # noqa: E402
    CONTROLLER_LIMITS_DEG,
    POSITION_MODE_SETTLE_S,
    go_home,
    home_warnings,
)

DEMO_START_JOINTS_DEG = PIPER_START_JOINTS_DEG["quest_0"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--joints-deg", type=float, nargs=6, metavar="DEG", default=DEMO_START_JOINTS_DEG,
                        help="Target J1..J6 in degrees (default: %(default)s)")
    target.add_argument("--home", choices=sorted(PIPER_START_JOINTS_DEG),
                        help="Named start from PIPER_START_JOINTS_DEG instead of --joints-deg")
    parser.add_argument("--speed-deg-s", type=float, default=10.0,
                        help="Speed of the fastest-moving joint (default: %(default)s)")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--stay", action="store_true",
                        help="Leave the arm holding the target instead of offering to move it back")
    parser.add_argument("--interface", default=DEFAULT_PIPER_INTERFACE)
    parser.add_argument("--channel", default=DEFAULT_PIPER_CHANNEL)
    parser.add_argument("--firmware", choices=("default", "v183", "v188"), default=DEFAULT_PIPER_FIRMWARE)
    parser.add_argument("--installation-pos", choices=("unchanged", "horizontal", "left", "right"),
                        default=DEFAULT_PIPER_INSTALLATION_POS)
    parser.add_argument("--payload", choices=("unchanged", "empty", "half", "full"), default=DEFAULT_PIPER_PAYLOAD)
    args = parser.parse_args(argv)
    if args.home:
        args.joints_deg = PIPER_START_JOINTS_DEG[args.home]
    if not all(math.isfinite(value) for value in args.joints_deg):
        parser.error("--joints-deg requires six finite values")
    for joint, (angle, (low, high)) in enumerate(zip(args.joints_deg, CONTROLLER_LIMITS_DEG), start=1):
        if not low <= angle <= high:
            parser.error(f"J{joint} = {angle:g} deg is outside the controller limits {low}..{high}")
    for name in ("speed_deg_s", "rate_hz"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.speed_deg_s > 30:
        parser.error("--speed-deg-s above 30 is not allowed for an in-person check")
    return args


def describe(api):
    joints = np.degrees(read_joint_angles(api))
    pose = api.get_eef_pose()
    print("  joints (deg): " + "  ".join(f"J{i+1} {value:7.2f}" for i, value in enumerate(joints)))
    if pose is not None:
        print(f"  flange xyz (mm): {np.round(np.multiply(pose[:3], 1000), 1).tolist()}  "
              f"roll/pitch/yaw (deg): {np.round(np.degrees(pose[3:]), 1).tolist()}")
    return joints


def move(api, target_deg, args):
    start = np.degrees(read_joint_angles(api))
    travel = np.max(np.abs(np.asarray(target_deg) - start))
    print(f"Largest joint travel {travel:.1f} deg, about {travel / args.speed_deg_s + 1:.0f} s at "
          f"{args.speed_deg_s:g} deg/s.")
    go_home(api, np.radians(target_deg), args.speed_deg_s, args.rate_hz)


def main(argv=None):
    args = parse_args(argv)
    for warning in home_warnings(args.joints_deg):
        print(f"WARNING: target {warning}")
    api = PiperAPI(firmware=args.firmware, interface=args.interface, channel=args.channel,
                   installation_pos=args.installation_pos, payload=args.payload)
    api.connect()
    try:
        if not api.is_ok():
            raise RuntimeError("Piper is not responding on the CAN bus")
        print("Current:")
        start_deg = describe(api)
        print("Target joints (deg): " + "  ".join(f"J{i+1} {value:7.2f}" for i, value in enumerate(args.joints_deg)))
        input("Clear the workspace, keep a hand near the e-stop, and press Enter to move (Ctrl-C to abort)...")
        if not api.enable():
            raise RuntimeError("Timed out while enabling Piper")
        api.set_position_mode()
        time.sleep(POSITION_MODE_SETTLE_S)
        api.set_speed_percent(100)  # The ramp in go_home sets the actual speed.
        move(api, args.joints_deg, args)
        print("Reached target:")
        describe(api)
        if not args.stay:
            answer = input("Press Enter to move back to the starting joints, or type 'stay' to keep holding: ")
            if answer.strip().lower() != "stay":
                move(api, start_deg, args)
                print("Back at the starting joints:")
                describe(api)
    finally:
        # Disconnecting does not disable the arm: it keeps holding its last position.
        api.disconnect()
    print("Piper is enabled and holding its position.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
