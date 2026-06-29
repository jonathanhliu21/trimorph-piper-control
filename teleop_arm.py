#!/usr/bin/env python3
"""Teleoperate the Piper arm from the live Teledex phone pose."""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


sys.path.insert(0, str(Path(__file__).resolve().parent))


TRANSLATION_SCALE = 0.5

HOME_POSE = [
    0.325,  # x [m]
    -0.3,  # y [m]
    0.2,  # z [m]
    -math.pi / 2.0,  # roll [rad]
    -0.659612,  # pitch [rad]
    -math.pi / 2.0,  # yaw [rad]
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Move Piper home, connect to an already-zeroed Teledex pose, then stream "
            "T_home * transformed_T_phone as the end-effector target."
        )
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=60.0,
        help="Target command rate in Hz. Default: 60",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=TRANSLATION_SCALE,
        help=f"Phone translation scale. Default: {TRANSLATION_SCALE}",
    )
    parser.add_argument(
        "--speed-percent",
        type=int,
        default=15,
        help="Arm speed percent during teleop. Default: 15",
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
        "--payload",
        choices=("empty", "half", "full", "unchanged"),
        default="half",
        help="Payload profile sent on connect. Use 'unchanged' to skip it. Default: half",
    )
    parser.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=3.0,
        help="Maximum seconds to wait for Piper OK status after connect. Default: 3",
    )
    parser.add_argument(
        "--no-enable",
        action="store_true",
        help="Do not send enable before moving.",
    )
    parser.add_argument(
        "--hand",
        action="store_true",
        help="Use the hand-tracked Teledex pose instead of the phone pose.",
    )
    parser.add_argument(
        "--return-home",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Command home when teleop exits. Default: true",
    )
    return parser.parse_args()


def validate_args(args):
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    if not math.isfinite(args.scale):
        raise ValueError("--scale must be finite")
    if not 0 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent must be between 0 and 100")
    if not 0 <= args.home_speed_percent <= 100:
        raise ValueError("--home-speed-percent must be between 0 and 100")
    if args.home_timeout_seconds <= 0:
        raise ValueError("--home-timeout-seconds must be positive")
    if args.home_tolerance_mm <= 0:
        raise ValueError("--home-tolerance-mm must be positive")
    if args.connect_timeout_seconds <= 0:
        raise ValueError("--connect-timeout-seconds must be positive")


def eef6_to_matrix(pose):
    values = [float(value) for value in pose]
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError("EEF pose must contain exactly 6 finite values")

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = values[:3]
    matrix[:3, :3] = R.from_euler("xyz", values[3:]).as_matrix()
    return matrix


def matrix_to_eef6(matrix):
    rotation = R.from_matrix(matrix[:3, :3]).as_euler("xyz")
    translation = matrix[:3, 3]
    return [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(rotation[0]),
        float(rotation[1]),
        float(rotation[2]),
    ]


def phone_pose_matrix(phone):
    position, rotation = phone.get_position_rotation()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return matrix


# teledex synthesizes position_hand / rotation_hand from the phone pose via this
# fixed phone-frame offset (see teledex.Session._process_data).
TELEDEX_PHONE_TO_HAND_ROTATION = R.from_euler("z", 180.0, degrees=True).as_matrix()


def teledex_frame_rotation(*, hand=False):
    # R_home^T cancels the R_home factor in home_matrix @ robot_pose,
    # giving target_pos = home_pos + delta_pos * scale (no axis coupling).
    home_R = R.from_euler("xyz", HOME_POSE[3:]).as_matrix()
    # R_x(-90°): world Z → Piper Y, world Y → Piper -Z
    phone_rotation = home_R.T @ R.from_euler("y", -135.0, degrees=True).as_matrix() @ R.from_euler("x", -90.0, degrees=True).as_matrix()
    if hand:
        return phone_rotation @ TELEDEX_PHONE_TO_HAND_ROTATION
    return phone_rotation


def transformed_phone_pose(phone_pose, scale, *, hand=False):
    frame_rotation = teledex_frame_rotation(hand=hand)
    frame_transform = np.eye(4, dtype=np.float64)
    frame_transform[:3, :3] = frame_rotation

    robot_pose = frame_transform @ phone_pose @ np.linalg.inv(frame_transform)
    robot_pose[:3, 3] *= scale
    return robot_pose


def format_pose(pose):
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    return ", ".join(f"{label}={value:+.6f}" for label, value in zip(labels, pose))


def format_piper_state(state):
    fields = [
        "is_ok",
        "receive_fps",
        "ctrl_mode",
        "arm_status",
        "mode_feedback",
        "teach_status",
        "motion_status",
        "err_code",
    ]
    return ", ".join(f"{field}={state.get(field, '')}" for field in fields)


def wait_for_piper_ok(api, *, timeout_s):
    start_t = time.monotonic()
    next_print_t = start_t
    last_state = api.get_state()

    while True:
        if api.is_ok():
            return True

        now = time.monotonic()
        if now >= next_print_t:
            print(f"Piper status: {format_piper_state(last_state)}")
            next_print_t = now + 0.5

        if now - start_t >= timeout_s:
            return False

        time.sleep(0.05)
        last_state = api.get_state()


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


def command_home(api, args, *, wait):
    print(f"Commanding home pose: {format_pose(HOME_POSE)}")
    api.set_speed_percent(args.home_speed_percent)
    api.move_eef(HOME_POSE)

    if wait:
        wait_until_xyz_reached(
            api,
            HOME_POSE,
            timeout_s=args.home_timeout_seconds,
            tolerance_mm=args.home_tolerance_mm,
        )
        print("Home reached.")


def format_target_line(tick, target_pose):
    return (
        f"target tick={tick:05d} "
        f"x={target_pose[0]:+.6f} y={target_pose[1]:+.6f} z={target_pose[2]:+.6f} "
        f"roll={target_pose[3]:+.4f} pitch={target_pose[4]:+.4f} "
        f"yaw={target_pose[5]:+.4f}"
    )


def stream_teleop(api, phone, args):
    dt = 1.0 / args.rate_hz
    home_matrix = eef6_to_matrix(HOME_POSE)

    print(
        f"Teleop streaming at {args.rate_hz:.1f} Hz, "
        f"translation scale={args.scale:.3f}."
    )
    print("Press the primary phone button or Ctrl+C to stop.")

    initial_T = phone_pose_matrix(phone)
    initial_R = initial_T[:3, :3].copy()
    initial_p = initial_T[:3, 3].copy()

    start_t = time.monotonic()
    tick = 0
    while True:
        if phone.get_button_state():
            print("Primary phone button pressed. Stopping teleop.")
            return

        T_current = phone_pose_matrix(phone)
        delta_matrix = np.eye(4, dtype=np.float64)
        delta_matrix[:3, :3] = initial_R.T @ T_current[:3, :3]
        if args.hand:
            delta_matrix[:3, 3] = initial_R.T @ (T_current[:3, 3] - initial_p)
        else:
            delta_matrix[:3, 3] = T_current[:3, 3] - initial_p
        robot_phone_pose = transformed_phone_pose(
            delta_matrix, args.scale, hand=args.hand
        )
        target_matrix = home_matrix @ robot_phone_pose
        target_pose = matrix_to_eef6(target_matrix)

        api.move_eef(target_pose)
        print(format_target_line(tick, target_pose), flush=True)

        tick += 1
        next_t = start_t + tick * dt
        sleep_s = next_t - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


def stop_phone(phone):
    stop = getattr(phone.session, "stop", None)
    if callable(stop):
        stop()


def main():
    args = parse_args()
    validate_args(args)

    from phone import Phone
    from scripts.piper_api import PiperAPI

    phone = None

    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
        installation_pos=args.installation_pos,
        payload=args.payload,
    )
    api.connect()
    time.sleep(0.3)

    try:
        if not wait_for_piper_ok(api, timeout_s=args.connect_timeout_seconds):
            state = api.get_state()
            raise RuntimeError(
                "Piper is not reporting OK after connect: "
                f"{format_piper_state(state)}. "
                "If receive_fps is 0, check CAN/power. If err_code or arm_status "
                "shows a fault, run clear_piper_error.py. If only the payload write "
                "is failing, retry with --payload unchanged."
            )

        if not args.no_enable:
            print("Enabling Piper...")
            if not api.enable(timeout_s=5.0):
                state = api.get_state()
                raise RuntimeError(
                    "Timed out while enabling Piper: "
                    f"{format_piper_state(state)}"
                )

        command_home(api, args, wait=True)

        phone = Phone()
        if args.hand:
            phone.hand = 1
            phone.position_name = "position_hand"
            phone.rotation_name = "rotation_hand"

        api.set_speed_percent(args.speed_percent)
        stream_teleop(api, phone, args)
        if args.return_home:
            command_home(api, args, wait=True)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        if args.return_home:
            command_home(api, args, wait=False)
    finally:
        if phone is not None:
            stop_phone(phone)
        api.disconnect()


if __name__ == "__main__":
    main()
