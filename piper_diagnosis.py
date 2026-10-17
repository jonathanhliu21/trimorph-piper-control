#!/usr/bin/env python3
"""Read Piper firmware, controller limits, and live status without moving it."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

from pyAgxArm import ArmModel, create_agx_arm_config

from scripts.piper_api.piper_api import PiperAPI


EEF_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
SOFTWARE_VERSION_RE = re.compile(r"S-V1\.8-(\d+)")


def expected_driver(software_version: str) -> str | None:
    match = SOFTWARE_VERSION_RE.search(software_version)
    if match is None:
        return None
    release = int(match.group(1))
    return "v188" if release >= 8 else "v183" if release >= 3 else "default"


def first_episode_pose(path: Path) -> list[float]:
    csv_path = path / "piper.csv" if path.is_dir() else path
    with csv_path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream), None)
    if row is None:
        raise ValueError(f"No Piper samples in {csv_path}")
    return [float(row[field]) for field in EEF_FIELDS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "episode",
        nargs="?",
        type=Path,
        help="Optional episode directory or piper.csv to inspect its first pose.",
    )
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can0")
    parser.add_argument(
        "--firmware", choices=("default", "v183", "v188"), default="v183",
        help="SDK driver selected by replay_episode.py (default: %(default)s).",
    )
    parser.add_argument("--timeout-s", type=float, default=1.0)
    return parser.parse_args()


def print_diagnosis(api: PiperAPI, *, episode: Path | None, timeout_s: float) -> None:
    robot = api.robot
    config = robot.get_config()
    configured_model = config.get("robot")
    print(f"SDK model selected by this repo: {configured_model}")
    print(f"SDK driver selected: {api.firmware}")
    if configured_model == ArmModel.PIPER:
        print(
            "NOTE: This repo selects generic PIPER. If your hardware is PiperX, "
            "its SDK model and preset joint limits differ."
        )

    firmware = api.get_firmware(timeout_s=timeout_s)
    if firmware is None:
        print("Controller firmware: no response")
    else:
        for key in (
            "software_version", "hardware_version", "node_type", "production_date"
        ):
            print(f"Controller {key}: {firmware.get(key, 'unknown')}")
        software_version = str(firmware.get("software_version", ""))
        expected = expected_driver(software_version)
        if expected is None:
            print("Firmware match: unknown version format; check SDK driver manually")
        else:
            verdict = "MATCH" if expected == api.firmware else "MISMATCH"
            print(f"Firmware match: {verdict} (expected SDK driver {expected})")

    state = api.get_state()
    print(f"CAN receive FPS: {state.get('receive_fps')}")
    for field in ("ctrl_mode", "arm_status", "mode_feedback", "teach_status", "motion_status", "err_code"):
        print(f"{field}: {state.get(field)}")
    if "TEACHING_MODE" in str(state.get("ctrl_mode", "")):
        print(
            "NOTE: The arm is in drag-teach mode. Manually moved joints can "
            "sit outside command limits; inspect the limits before replay."
        )
    print(f"motors enabled: {robot.get_joints_enable_status_list()}")
    print(f"SDK joint-limit clamp enabled: {robot.get_joint_limits_enabled()}")

    preset = create_agx_arm_config(robot=ArmModel.PIPER_X)["joint_limits"]
    generic_preset = create_agx_arm_config(robot=ArmModel.PIPER)["joint_limits"]
    print("Controller joint limits (rad), live joint angle (rad), PiperX SDK preset (rad):")
    missing = []
    suspicious = []
    generic_matches = []
    outside_current_limits = []
    unreadable_angles = []
    invalid_limits = []
    for joint in range(1, 7):
        name = f"joint{joint}"
        expected_min, expected_max = preset[name]
        feedback = robot.get_joint_angle_vel_limits(
            joint, timeout=timeout_s, min_interval=0.0
        )
        angle_text = state.get(f"joint_{joint}", "")
        if feedback is None:
            print(f"  J{joint}: controller did not return limits; angle={angle_text}")
            missing.append(joint)
            continue
        limits = feedback.msg
        low = float(limits.min_angle_limit)
        high = float(limits.max_angle_limit)
        notes = []
        generic_min, generic_max = generic_preset[name]
        if abs(low - generic_min) < 0.001 and abs(high - generic_max) < 0.001:
            generic_matches.append(joint)
        if not all(math.isfinite(value) for value in (low, high)) or high <= low:
            notes.append("INVALID LIMIT RANGE")
            invalid_limits.append(joint)
        elif low > expected_min + 0.02 or high < expected_max - 0.02:
            notes.append("narrower than PiperX SDK preset")
        try:
            angle = float(angle_text)
            if joint not in invalid_limits and (angle < low or angle > high):
                notes.append("CURRENT ANGLE OUTSIDE CONTROLLER LIMITS")
                outside_current_limits.append(joint)
            elif joint not in invalid_limits and min(angle - low, high - angle) < math.radians(2):
                margin_deg = math.degrees(min(angle - low, high - angle))
                notes.append(f"NEAR CONTROLLER LIMIT ({margin_deg:.3f} deg margin)")
        except (TypeError, ValueError):
            unreadable_angles.append(joint)
        if notes:
            suspicious.append(joint)
        print(
            f"  J{joint}: controller=[{low:+.6f}, {high:+.6f}] "
            f"angle={angle_text} preset=[{expected_min:+.6f}, {expected_max:+.6f}] "
            f"max_speed={limits.max_joint_spd} "
            + ("; ".join(notes) if notes else "OK")
        )

    if missing:
        print(f"Limit readback unavailable for joints {missing}; check CAN replies.")
    if {4, 5}.issubset(generic_matches):
        print(
            "Controller J4/J5 limits match the generic PIPER SDK preset; "
            "both differ from the PiperX preset. This does not identify the "
            "physical model or prove which IK joint target was rejected."
        )
    if suspicious:
        print(
            f"Review controller limits for joints {suspicious}. Narrower limits "
            "may be intentional; do not change them without checking the hardware."
        )
    if outside_current_limits:
        print(
            "Current joint-limit precheck: FAIL; measured joints outside "
            f"controller angle limits: {outside_current_limits}"
        )
    elif missing or unreadable_angles or invalid_limits:
        print(
            "Current joint-limit precheck: UNKNOWN; limits missing for "
            f"{missing}, angles unreadable for {unreadable_angles}, "
            f"invalid limit ranges for {invalid_limits}"
        )
    else:
        print("Current joint-limit precheck: PASS; all measured joints are within limits")
    print(
        "MOVE_P acceptance at the current arm position: UNDETERMINED. "
        "This firmware provides no read-only IK target check; arm_status may "
        "describe a previous rejected command."
    )
    if episode is not None:
        target = first_episode_pose(episode)
        print("Episode first EEF pose [x,y,z m; roll,pitch,yaw rad]:")
        print("  " + ", ".join(f"{value:+.6f}" for value in target))
        try:
            measured = [float(state[field]) for field in EEF_FIELDS]
        except (TypeError, ValueError):
            print("Current EEF pose unavailable")
        else:
            distance = math.dist(target[:3], measured[:3])
            print(f"Current-to-first-target position distance: {distance:.3f} m")
            print(
                "Episode first-target acceptance: UNDETERMINED without sending "
                "that target; a passing current joint-limit precheck cannot "
                "establish that its IK solution is within limits."
            )


def main() -> int:
    args = parse_args()
    if args.timeout_s <= 0:
        print("error: --timeout-s must be positive", file=sys.stderr)
        return 2
    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
        installation_pos=None,
        payload=None,
    )
    try:
        api.connect()
        print_diagnosis(api, episode=args.episode, timeout_s=args.timeout_s)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        api.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
