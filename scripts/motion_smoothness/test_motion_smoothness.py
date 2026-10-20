#!/usr/bin/env python3
"""Replay deterministic smoothness trajectories on selected finger joints.

The command loop intentionally matches ``data_collection/replay_episode.py``:
an analytic trajectory is sampled at ``--rate-hz`` using episode time, and each
position-only target is sent with ``hand.set_positions(..., clip=True)``. No
desired velocity or acceleration is calculated from the trajectory.

Run from the repository root, for example:

    .venv/bin/python scripts/motion_smoothness/test_motion_smoothness.py \
        --motion slow-sweep --joint 4
    .venv/bin/python scripts/motion_smoothness/test_motion_smoothness.py \
        --motion triangle --joint 5
    .venv/bin/python scripts/motion_smoothness/test_motion_smoothness.py \
        --motion sinusoid --joint 4 5

Review the motion constants and ``config/one_finger_test.yaml`` before moving
hardware. Torque is disabled during shutdown, including after an interruption.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make repository packages available during direct script execution.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import yaml

from data_collection.collection.hand_common import is_compatible_dynamixel_model
from midas_hand_api import HandConfig, MidasHand

DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "config" / "one_finger_test.yaml"
MOTOR_IDS = (4, 5)  # PIP, MCP flexion

# Replay settings match data_collection/replay_episode.py.
REPLAY_HZ = 30.0
MOTION_DURATION_S = 10.0
PREPOSITION_SECONDS = 3.0
SETTLE_SECONDS = 1.0
PROFILE_MOVE_VELOCITY_RAD_S = 3.0
PROFILE_MOVE_ACCELERATION_RAD_S2 = 50.0

# Tunable motion amplitudes. All three tests keep unselected joints straight at
# 0 rad. Selected joints follow the same trajectory in phase. Defaults use less
# than the full [-1.95, 0.0] physical range.
SLOW_SWEEP_AMPLITUDE_RAD = 1.50
TRIANGLE_AMPLITUDE_RAD = 1.50
SINUSOID_AMPLITUDE_RAD = 0.75
SINUSOID_CENTER_RAD = -0.75
SINUSOID_CYCLES = 1.0

MOTION_NAMES = ("slow-sweep", "triangle", "sinusoid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", choices=MOTION_NAMES, required=True)
    parser.add_argument(
        "--joint",
        type=int,
        choices=MOTOR_IDS,
        nargs="+",
        required=True,
        help=(
            "Joints to move: motor 4 (PIP), motor 5 (MCP flexion), or both "
            "(for example: --joint 4 5)"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Hand YAML configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=REPLAY_HZ,
        help=f"Position command rate (default: {REPLAY_HZ:g} Hz)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the physical-motion confirmation prompt",
    )
    args = parser.parse_args()
    if args.rate_hz <= 0:
        parser.error("--rate-hz must be greater than 0")
    if len(set(args.joint)) != len(args.joint):
        parser.error("--joint values must be unique")
    if not args.config.is_file():
        parser.error(f"configuration file does not exist: {args.config}")
    return args


def load_config(path: Path) -> HandConfig:
    """Load exactly motors 4 and 5 and reject an unreviewed configuration."""

    with path.open(encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream) or {}
    if raw_config.get("placeholders_unreviewed", True):
        raise RuntimeError(
            "Motion refused: review the configuration and set "
            "placeholders_unreviewed: false"
        )

    config = HandConfig.load(path).subset(MOTOR_IDS)
    if tuple(config.motor_ids) != MOTOR_IDS:
        raise ValueError(
            f"Expected configured motors {list(MOTOR_IDS)}, got "
            f"{list(config.motor_ids)}"
        )
    return config


def sample_motion(
    motion: str, joints: tuple[int, ...], query_t: float
) -> np.ndarray:
    """Evaluate a deterministic position-only trajectory at episode time.

    Unlike ``sample_stream`` in episode replay, this function does not
    interpolate recorded waypoints. It directly evaluates the selected
    analytic function while returning positions in configured motor order
    ``(4, 5)``.
    """

    if motion not in MOTION_NAMES:
        raise ValueError(f"Unknown motion: {motion}")
    if not joints:
        raise ValueError("At least one joint is required")
    unknown_joints = sorted(set(joints) - set(MOTOR_IDS))
    if unknown_joints:
        raise ValueError(f"Joints must be selected from {MOTOR_IDS}: {unknown_joints}")
    if len(set(joints)) != len(joints):
        raise ValueError("Joints must be unique")

    t = float(np.clip(query_t, 0.0, MOTION_DURATION_S))
    phase = t / MOTION_DURATION_S

    if motion == "slow-sweep":
        moving_position = -SLOW_SWEEP_AMPLITUDE_RAD * phase
    elif motion == "triangle":
        # 0 -> -amplitude -> 0, with the reversal halfway through the test.
        moving_position = -TRIANGLE_AMPLITUDE_RAD * (1.0 - abs(2.0 * phase - 1.0))
    else:
        # Starts and ends straight for the default center == -amplitude.
        angle = 2.0 * np.pi * SINUSOID_CYCLES * phase
        moving_position = SINUSOID_CENTER_RAD + SINUSOID_AMPLITUDE_RAD * np.cos(angle)

    positions = np.zeros(len(MOTOR_IDS), dtype=np.float64)
    for joint in joints:
        positions[MOTOR_IDS.index(joint)] = moving_position
    return positions


def validate_trajectory(
    config: HandConfig, motion: str, joints: tuple[int, ...]
) -> None:
    """Check the full analytic trajectory against YAML limits before motion."""

    # This dense deterministic check includes both endpoints and the triangle
    # reversal. It is independent of the selected command rate.
    check_times = np.linspace(0.0, MOTION_DURATION_S, 10_001)
    positions = np.asarray(
        [sample_motion(motion, joints, float(t)) for t in check_times]
    )
    lower = config.joint_lower_limits_array
    upper = config.joint_upper_limits_array
    below = positions < lower
    above = positions > upper
    if np.any(below | above):
        bad_sample, bad_joint_index = np.argwhere(below | above)[0]
        motor_id = config.motor_ids[int(bad_joint_index)]
        target = positions[int(bad_sample), int(bad_joint_index)]
        raise ValueError(
            f"Generated motor {motor_id} target {target:.3f} rad is outside "
            f"configured limits [{lower[bad_joint_index]:.3f}, "
            f"{upper[bad_joint_index]:.3f}]"
        )

    minima = positions.min(axis=0)
    maxima = positions.max(axis=0)
    for index, motor_id in enumerate(config.motor_ids):
        print(
            f"Motor {motor_id}: generated [{minima[index]:.3f}, "
            f"{maxima[index]:.3f}] rad; YAML limits "
            f"[{lower[index]:.3f}, {upper[index]:.3f}] rad"
        )


def verify_bus(hand: MidasHand) -> None:
    models = hand.ping()
    missing = sorted(set(hand.motor_ids) - set(models))
    wrong_models = {
        motor_id: model_number
        for motor_id, model_number in models.items()
        if not is_compatible_dynamixel_model(
            model_number, hand.config.expected_model_number
        )
    }
    if missing:
        raise OSError(f"Configured motors did not respond: {missing}")
    if wrong_models:
        raise OSError(f"Unexpected Dynamixel model numbers: {wrong_models}")


def confirm_motion(args: argparse.Namespace) -> None:
    if args.yes:
        return
    motor_list = ", ".join(str(joint) for joint in args.joint)
    response = input(
        f"{args.motion} on motor(s) {motor_list} will move physical hardware. "
        "Type MOVE to continue: "
    ).strip()
    if response != "MOVE":
        raise RuntimeError("Motion cancelled")


def move_to_first_pose(hand: MidasHand, first_pose: np.ndarray) -> None:
    """Use the same repeated preposition command pattern as episode replay."""

    print("Moving to first pose...")
    end_t = time.monotonic() + PREPOSITION_SECONDS
    while time.monotonic() < end_t:
        hand.set_positions(first_pose, clip=True)
        time.sleep(0.05)
    if SETTLE_SECONDS > 0:
        time.sleep(SETTLE_SECONDS)


def replay_motion(
    hand: MidasHand,
    motion: str,
    joints: tuple[int, ...],
    rate_hz: float,
) -> None:
    """Run the same episode-time scheduling loop as replay_episode.py."""

    period_s = 1.0 / rate_hz
    wall_start = time.monotonic()
    tick = 0
    motor_list = ", ".join(str(joint) for joint in joints)
    print(
        f"Replaying {motion} on motor(s) {motor_list} for "
        f"{MOTION_DURATION_S:.1f}s "
        f"at {rate_hz:.1f} Hz."
    )
    while True:
        episode_t = min(tick * period_s, MOTION_DURATION_S)
        hand.set_positions(sample_motion(motion, joints, episode_t), clip=True)
        if episode_t >= MOTION_DURATION_S:
            break
        tick += 1
        sleep_s = wall_start + tick * period_s - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
    print("Replay complete.")


def main() -> int:
    args = parse_args()
    joints = tuple(args.joint)
    config = load_config(args.config)
    validate_trajectory(config, args.motion, joints)
    confirm_motion(args)

    hand: MidasHand | None = None
    try:
        hand = MidasHand(config=config, autoconnect=True)
        verify_bus(hand)
        hand.configure(enable_torque=False)
        hand.set_motion_profile(
            profile_velocity_rad_s=PROFILE_MOVE_VELOCITY_RAD_S,
            profile_acceleration_rad_s2=PROFILE_MOVE_ACCELERATION_RAD_S2,
        )
        # As in episode replay, establish a non-jumping goal before torque-on.
        hand.set_positions(hand.read_pos(), clip=True)
        hand.enable_torque()
        print(f"TRIMORPH hand connected on {hand.port}")

        first_pose = sample_motion(args.motion, joints, 0.0)
        move_to_first_pose(hand, first_pose)
        replay_motion(hand, args.motion, joints, args.rate_hz)
        return 0
    except KeyboardInterrupt:
        print("\nReplay interrupted.")
        return 130
    finally:
        if hand is not None:
            hand.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
