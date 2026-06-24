#!/usr/bin/env python3
"""Stream a smooth YZ-plane circle to the Piper end effector."""

import argparse
import math
import sys
import time
from pathlib import Path


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
        description=(
            "Move Piper home, then stream a circular end-effector target in "
            "the YZ plane. Home is the top of the circle."
        )
    )
    parser.add_argument(
        "--radius-mm",
        type=float,
        default=50.0,
        help="Circle radius in millimeters. Default: 50",
    )
    parser.add_argument(
        "--period-seconds",
        type=float,
        default=4.0,
        help="Seconds per circle for the moving target. Default: 4",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=60.0,
        help="Target command rate in Hz. Default: 60",
    )
    parser.add_argument(
        "--cycles",
        type=float,
        default=3.0,
        help="Number of circles to stream. Ignored with --continuous. Default: 3",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run circles until Ctrl+C.",
    )
    parser.add_argument(
        "--speed-percent",
        type=int,
        default=15,
        help="Arm speed percent. Lower values make the arm lag the target. Default: 15",
    )
    parser.add_argument(
        "--home-speed-percent",
        type=int,
        default=20,
        help="Arm speed percent for the initial home move. Default: 20",
    )
    parser.add_argument(
        "--home-timeout-seconds",
        type=float,
        default=10.0,
        help="Maximum seconds to wait for home. Default: 10",
    )
    parser.add_argument(
        "--home-tolerance-mm",
        type=float,
        default=2.0,
        help="XYZ norm tolerance for reaching home in millimeters. Default: 2",
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
        "--return-home",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Command home after the circle stream ends. Default: true",
    )
    return parser.parse_args()


def validate_args(args):
    if args.radius_mm <= 0:
        raise ValueError("--radius-mm must be positive")
    if args.period_seconds <= 0:
        raise ValueError("--period-seconds must be positive")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    if args.cycles <= 0 and not args.continuous:
        raise ValueError("--cycles must be positive unless --continuous is set")
    if not 0 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent must be between 0 and 100")
    if not 0 <= args.home_speed_percent <= 100:
        raise ValueError("--home-speed-percent must be between 0 and 100")
    if args.home_timeout_seconds <= 0:
        raise ValueError("--home-timeout-seconds must be positive")
    if args.home_tolerance_mm <= 0:
        raise ValueError("--home-tolerance-mm must be positive")


def format_pose(pose):
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    return ", ".join(f"{label}={value:+.6f}" for label, value in zip(labels, pose))


def xyz_error_norm_mm(target_pose, actual_pose):
    return math.sqrt(
        sum(((target_pose[i] - actual_pose[i]) * 1000.0) ** 2 for i in range(3))
    )


def wait_until_xyz_reached(api, target_pose, *, timeout_s, tolerance_mm):
    start_t = time.monotonic()
    next_print_t = start_t

    while True:
        actual_pose = api.get_eef_pose()
        if actual_pose is None:
            raise RuntimeError("Could not read current end-effector pose")

        error_norm_mm = xyz_error_norm_mm(target_pose, actual_pose)
        now = time.monotonic()

        if now >= next_print_t:
            print(
                f"Home wait: XYZ error norm {error_norm_mm:.3f} mm "
                f"(target < {tolerance_mm:.3f} mm)"
            )
            next_print_t = now + 0.5

        if error_norm_mm < tolerance_mm:
            return
        if now - start_t >= timeout_s:
            raise RuntimeError(
                f"Timed out after {timeout_s:.1f}s waiting for home "
                f"XYZ error norm < {tolerance_mm:.3f} mm"
            )

        time.sleep(0.05)


def circle_pose(home_pose, radius_m, theta):
    target_pose = list(home_pose)

    # Home is the top of the circle. Positive theta moves first toward -Y:
    # top -> negative Y side -> lower Z side -> positive Y side -> top.
    target_pose[1] = home_pose[1] - radius_m * math.sin(theta)
    target_pose[2] = home_pose[2] - radius_m + radius_m * math.cos(theta)
    return target_pose


def format_target_line(tick, elapsed_s, theta, target_pose):
    return (
        f"target tick={tick:05d} t={elapsed_s:8.4f}s "
        f"theta={math.degrees(theta) % 360.0:8.3f}deg "
        f"x={target_pose[0]:+.6f} y={target_pose[1]:+.6f} z={target_pose[2]:+.6f} "
        f"dy={(target_pose[1] - HOME_POSE[1]) * 1000.0:+8.3f}mm "
        f"dz={(target_pose[2] - HOME_POSE[2]) * 1000.0:+8.3f}mm"
    )


def stream_circle(api, args):
    radius_m = args.radius_mm / 1000.0
    dt = 1.0 / args.rate_hz
    max_t = None if args.continuous else args.cycles * args.period_seconds

    print(
        f"Streaming YZ circle: radius={args.radius_mm:.1f} mm, "
        f"period={args.period_seconds:.3f}s, rate={args.rate_hz:.1f} Hz, "
        f"speed={args.speed_percent}%"
    )
    print("Direction: home/top -> -Y side -> lower Z side -> +Y side -> home/top.")

    start_t = time.monotonic()
    tick = 0
    while True:
        elapsed_s = time.monotonic() - start_t
        if max_t is not None and elapsed_s >= max_t:
            break

        theta = 2.0 * math.pi * (elapsed_s / args.period_seconds)
        target_pose = circle_pose(HOME_POSE, radius_m, theta)
        api.move_eef(target_pose)
        print(format_target_line(tick, elapsed_s, theta, target_pose), flush=True)

        tick += 1
        next_t = start_t + tick * dt
        sleep_s = next_t - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


def main():
    args = parse_args()
    validate_args(args)

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

        print(f"Commanding home pose: {format_pose(HOME_POSE)}")
        api.set_speed_percent(args.home_speed_percent)
        api.move_eef(HOME_POSE)
        wait_until_xyz_reached(
            api,
            HOME_POSE,
            timeout_s=args.home_timeout_seconds,
            tolerance_mm=args.home_tolerance_mm,
        )
        print("Home reached.")

        api.set_speed_percent(args.speed_percent)
        stream_circle(api, args)

        if args.return_home:
            print("Commanding home pose.")
            api.set_speed_percent(args.home_speed_percent)
            api.move_eef(HOME_POSE)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        if args.return_home:
            print("Commanding home pose.")
            api.set_speed_percent(args.home_speed_percent)
            api.move_eef(HOME_POSE)
    finally:
        api.disconnect()


if __name__ == "__main__":
    main()
