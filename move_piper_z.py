#!/usr/bin/env python3
"""Move the Piper end effector by a small relative XYZ offset."""

import argparse
import math
import sys
import time
from pathlib import Path
import teledex


sys.path.insert(0, str(Path(__file__).resolve().parent))


HOME_POSE = [
    0.325,  # x [m]
    -0.3,  # y [m]
    0.25,  # z [m]
    -math.pi / 2.0,  # roll [rad]
    -0.659612,  # pitch [rad]
    -math.pi / 2.0,  # yaw [rad]
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move the Piper end effector by a small relative XYZ offset."
    )
    parser.add_argument(
        "--home",
        action="store_true",
        help="Move to HOME_POSE instead of applying a relative XYZ move.",
    )
    parser.add_argument(
        "--dxyz-mm",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=None,
        help=(
            "Relative XYZ move in millimeters. Overrides --dx-mm, --dy-mm, "
            "and --dz-mm. Example: --dxyz-mm 0 0 5"
        ),
    )
    parser.add_argument(
        "--dx-mm",
        type=float,
        default=0.0,
        help="Relative X move in millimeters. Default: 0.0",
    )
    parser.add_argument(
        "--dy-mm",
        type=float,
        default=0.0,
        help="Relative Y move in millimeters. Default: 0.0",
    )
    parser.add_argument(
        "--dz-mm",
        type=float,
        default=5.0,
        help="Relative Z move in millimeters. Default: 5.0",
    )
    parser.add_argument(
        "--speed-percent",
        type=int,
        default=20,
        help="Arm speed percent for the move. Default: 20",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="Seconds to wait after sending the move command. Default: 2",
    )
    parser.add_argument(
        "--interface",
        default="socketcan",
        help="python-can interface. Default: socketcan",
    )
    parser.add_argument(
        "--channel",
        default="can0",
        help="CAN channel. Default: can0",
    )
    parser.add_argument(
        "--firmware",
        choices=("default", "v183", "v188"),
        default="default",
        help="Piper firmware driver selection. Default: default",
    )
    parser.add_argument(
        "--installation-pos",
        choices=("horizontal", "left", "right", "unchanged"),
        default="right",
        help="Installation profile sent on connect. Default: right",
    )
    parser.add_argument(
        "--no-enable",
        action="store_true",
        help="Do not send enable before moving.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="Maximum seconds to wait for the target. Default: 10",
    )
    parser.add_argument(
        "--tolerance-mm",
        type=float,
        default=1.0,
        help="Success threshold for XYZ norm-to-target in millimeters. Default: 1.0",
    )
    return parser.parse_args()


def format_pose(pose):
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    return ", ".join(f"{label}={value:+.6f}" for label, value in zip(labels, pose))


def format_delta_mm(delta_mm):
    labels = ["dx", "dy", "dz"]
    return ", ".join(f"{label}={value:+.3f} mm" for label, value in zip(labels, delta_mm))


def get_delta_mm(args):
    delta_mm = args.dxyz_mm if args.dxyz_mm is not None else [
        args.dx_mm,
        args.dy_mm,
        args.dz_mm,
    ]
    if len(delta_mm) != 3:
        raise ValueError("Relative move must contain exactly 3 values")
    if not all(math.isfinite(value) for value in delta_mm):
        raise ValueError("Relative move values must be finite")
    return [float(value) for value in delta_mm]


def xyz_error_mm(target_pose, actual_pose):
    return [(target_pose[i] - actual_pose[i]) * 1000.0 for i in range(3)]


def xyz_error_norm_mm(error_mm):
    return math.sqrt(sum(value * value for value in error_mm))


def format_xyz_error(error_mm):
    return (
        f"dx={error_mm[0]:+.3f} mm, "
        f"dy={error_mm[1]:+.3f} mm, "
        f"dz={error_mm[2]:+.3f} mm, "
        f"norm={xyz_error_norm_mm(error_mm):.3f} mm"
    )


def wait_until_target_reached(api, args, target_pose):
    start_t = time.monotonic()
    next_print_t = start_t

    while True:
        actual_pose = api.get_eef_pose()
        if actual_pose is None:
            raise RuntimeError("Could not read current end-effector pose")

        error_mm = xyz_error_mm(target_pose, actual_pose)
        error_norm_mm = xyz_error_norm_mm(error_mm)
        now = time.monotonic()

        if now >= next_print_t:
            print(f"Actual pose:  {format_pose(actual_pose)}")
            print(f"XYZ error:    {format_xyz_error(error_mm)}")
            next_print_t = now + 0.5

        if error_norm_mm < args.tolerance_mm:
            print(
                f"Success: reached target with XYZ error norm "
                f"{error_norm_mm:.3f} mm < {args.tolerance_mm:.3f} mm."
            )
            return

        if now - start_t >= args.timeout_seconds:
            raise RuntimeError(
                f"Timed out after {args.timeout_seconds:.1f}s waiting for "
                f"XYZ error norm < {args.tolerance_mm:.3f} mm"
            )

        time.sleep(0.05)


def command_target_pose(api, args, target_pose):
    print(f"Target pose:  {format_pose(target_pose)}")
    api.move_eef(target_pose)

    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)

    wait_until_target_reached(api, args, target_pose)


def go_home(api, args):
    print("Commanding home pose.")
    command_target_pose(api, args, list(HOME_POSE))


def main():
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.tolerance_mm <= 0:
        raise ValueError("--tolerance-mm must be positive")

    from scripts.piper_api import PiperAPI

    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
        installation_pos=args.installation_pos,
    )
    api.connect()
    time.sleep(0.3)

    try:
        if not api.is_ok():
            raise RuntimeError("Piper is not reporting OK after connect")

        if not args.no_enable:
            print("Enabling Piper...")
            if not api.enable(timeout_s=5.0):
                raise RuntimeError("Timed out while enabling Piper")

        api.set_speed_percent(args.speed_percent)

        current_pose = api.get_eef_pose()
        if current_pose is None:
            raise RuntimeError("Could not read current end-effector pose")

        print(f"Current pose: {format_pose(current_pose)}")

        if args.home:
            go_home(api, args)
        else:
            delta_mm = get_delta_mm(args)
            delta_m = [value / 1000.0 for value in delta_mm]
            target_pose = list(current_pose)
            for i, value in enumerate(delta_m):
                target_pose[i] += value
            print(f"Commanding relative XYZ move: {format_delta_mm(delta_mm)}")
            command_target_pose(api, args, target_pose)

    finally:
        api.disconnect()


if __name__ == "__main__":
    main()
