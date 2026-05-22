import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.midas_api import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    FINGER_IDS,
    apply_finger_values,
    import_hand_api,
    load_hand_config,
)


# ---------------------------------------------------------------------------
# Edit these values, then run:
#   python3 examples/midas_hand/move_finger.py
# ---------------------------------------------------------------------------

# Each command is ["finger", values...].
# Fingers use 3 values. Thumb uses 4 values.
# A step can be one command, or a list of commands to move in parallel.
SEQUENCE = [
    [
        ["thumb", 0.0, 0.0, 0.0, 0.05],
        ["index", 0.0, 0.0, 0.0],
        ["middle", -0.05, -0.05, 0.0],
        ["ring", -0.05, -0.05, 0.0],
    ],
    ["index", -0.2791845, -0.40712627, 0.0],
]

# Seconds to wait after each commanded movement.
WAIT_BETWEEN_MOVEMENTS_S = 0.2

# Repeat the whole sequence this many times. Use None to repeat until Ctrl+C.
REPEAT_COUNT = 200
REPEAT_DELAY_S = 3.0

# Use "current" to keep non-commanded motors where they are, or "zero" to
# command non-commanded motors to zero while running the sequence.
BASE_POSE = "zero"

# Connection/config. Leave PORT as None to auto-detect.
PORT = None
BAUDRATE = 1_000_000
CONFIG_PATH = DEFAULT_CONFIG_PATH
OPERATING_MODE_POSITION = 3

# Motion behavior.
POSITION_P_GAIN = 800
POSITION_I_GAIN = 0
POSITION_D_GAIN = 400
GOAL_CURRENT_LIMIT_MA = None
WAIT_UNTIL_REACHED = True
REACH_TIMEOUT_S = 10.0
DEFAULT_REACH_TOLERANCE_RAD = 0.05
FINGER_REACH_TOLERANCE_RAD = {
    "index": 0.015,
}
ERROR_PRINT_INTERVAL_S = 0.25
PROFILE_VELOCITY_RAD_S = 0.4
PROFILE_ACCELERATION_RAD_S_2 = 1.0


class HandArgs:
    port = PORT
    baudrate = BAUDRATE
    config = str(CONFIG_PATH)


def connect_position_mode_hand(enable_torque=True):
    _, _, MidasHand = import_hand_api()
    config = replace(load_hand_config(HandArgs), operating_mode=OPERATING_MODE_POSITION)
    hand = MidasHand(config)
    hand.configure(enable_torque=enable_torque)
    return hand


def is_command(value):
    return isinstance(value, (list, tuple)) and value and isinstance(value[0], str)


def normalize_step(step):
    if is_command(step):
        commands = [step]
    elif isinstance(step, (list, tuple)) and step and all(is_command(item) for item in step):
        commands = list(step)
    else:
        raise ValueError(
            "Each SEQUENCE step must be ['finger', values...] or a list of those commands"
        )

    normalized = []
    seen_fingers = set()
    for command in commands:
        finger = command[0]
        if finger not in FINGER_IDS:
            raise ValueError(f"Unknown finger {finger!r}. Choose one of {sorted(FINGER_IDS)}")
        if finger in seen_fingers:
            raise ValueError(f"Finger {finger!r} appears more than once in one parallel step")
        seen_fingers.add(finger)

        values = np.asarray(command[1:], dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"{finger} command values must be a 1D vector")
        expected_dim = len(FINGER_IDS[finger])
        if len(values) != expected_dim:
            raise ValueError(f"{finger} expects {expected_dim} values, got {len(values)}")
        normalized.append((finger, values))
    return normalized


def normalize_sequence(sequence):
    if not isinstance(sequence, (list, tuple)) or not sequence:
        raise ValueError("SEQUENCE must be a non-empty list of steps")
    return [normalize_step(step) for step in sequence]


def describe_step(commands):
    return ", ".join(f"{finger}: {values.tolist()}" for finger, values in commands)


def apply_step(base, hand, commands):
    target = np.asarray(base, dtype=np.float64).copy()
    for finger, values in commands:
        target = apply_finger_values(target, hand, finger, values)
    return target


def finger_errors(hand, target, commands):
    current = hand.read_pos()
    parts = []
    for finger, _ in commands:
        motor_ids = FINGER_IDS[finger]
        indices = [hand.motor_ids.index(motor_id) for motor_id in motor_ids]
        error = target[indices] - current[indices]
        parts.append(f"{finger}: {np.round(error, 5).tolist()}")
    max_abs_error = float(np.max(np.abs(target - current)))
    return max_abs_error, " | ".join(parts)


def reach_tolerances(hand):
    tolerances = np.full(len(hand.motor_ids), DEFAULT_REACH_TOLERANCE_RAD)
    for finger, tolerance in FINGER_REACH_TOLERANCE_RAD.items():
        if finger not in FINGER_IDS:
            raise ValueError(f"Unknown finger {finger!r} in FINGER_REACH_TOLERANCE_RAD")
        for motor_id in FINGER_IDS[finger]:
            if motor_id in hand.motor_ids:
                tolerances[hand.motor_ids.index(motor_id)] = float(tolerance)
    return tolerances


def wait_until_reached_with_error_print(hand, target, commands):
    tolerances = reach_tolerances(hand)
    deadline = time.monotonic() + REACH_TIMEOUT_S
    next_print_t = 0.0
    while time.monotonic() < deadline:
        max_abs_error, error_text = finger_errors(hand, target, commands)
        current = hand.read_pos()
        within_tolerance = np.all(np.abs(target - current) <= tolerances)
        if time.monotonic() >= next_print_t:
            print(f"  error rad max={max_abs_error:.5f} | {error_text}")
            next_print_t = time.monotonic() + ERROR_PRINT_INTERVAL_S
        if within_tolerance:
            print(f"  reached: error rad max={max_abs_error:.5f} | {error_text}")
            return True
        time.sleep(0.02)

    max_abs_error, error_text = finger_errors(hand, target, commands)
    print(f"  final error rad max={max_abs_error:.5f} | {error_text}")
    return False


def main():
    if WAIT_BETWEEN_MOVEMENTS_S < 0:
        raise ValueError("WAIT_BETWEEN_MOVEMENTS_S must be 0 or greater")
    if REPEAT_COUNT is not None and REPEAT_COUNT <= 0:
        raise ValueError("REPEAT_COUNT must be greater than 0, or None")
    if REPEAT_DELAY_S < 0:
        raise ValueError("REPEAT_DELAY_S must be 0 or greater")
    if BASE_POSE not in ("current", "zero"):
        raise ValueError("BASE_POSE must be 'current' or 'zero'")
    if REACH_TIMEOUT_S < 0:
        raise ValueError("REACH_TIMEOUT_S must be 0 or greater")
    if DEFAULT_REACH_TOLERANCE_RAD < 0:
        raise ValueError("DEFAULT_REACH_TOLERANCE_RAD must be 0 or greater")
    for finger, tolerance in FINGER_REACH_TOLERANCE_RAD.items():
        if finger not in FINGER_IDS:
            raise ValueError(f"Unknown finger {finger!r} in FINGER_REACH_TOLERANCE_RAD")
        if tolerance < 0:
            raise ValueError("Finger reach tolerances must be 0 or greater")
    if ERROR_PRINT_INTERVAL_S <= 0:
        raise ValueError("ERROR_PRINT_INTERVAL_S must be greater than 0")

    sequence = normalize_sequence(SEQUENCE)

    hand = connect_position_mode_hand(enable_torque=True)
    try:
        print(f"Set operating mode: Position Control ({OPERATING_MODE_POSITION})")
        hand.set_gains(p=POSITION_P_GAIN, i=POSITION_I_GAIN, d=POSITION_D_GAIN)
        print(
            f"Set position gains: P={POSITION_P_GAIN}, "
            f"I={POSITION_I_GAIN}, D={POSITION_D_GAIN}"
        )
        if GOAL_CURRENT_LIMIT_MA is not None:
            hand.set_current_limit(GOAL_CURRENT_LIMIT_MA)
            print(f"Set goal current limit: {GOAL_CURRENT_LIMIT_MA} mA")

        if PROFILE_VELOCITY_RAD_S is not None or PROFILE_ACCELERATION_RAD_S_2 is not None:
            hand.set_motion_profile(
                profile_velocity_rad_s=PROFILE_VELOCITY_RAD_S,
                profile_acceleration_rad_s2=PROFILE_ACCELERATION_RAD_S_2,
            )

        print(f"Hand connected on {hand.port}")
        base = np.zeros(len(hand.motor_ids)) if BASE_POSE == "zero" else hand.read_pos()

        cycle = 0
        while REPEAT_COUNT is None or cycle < REPEAT_COUNT:
            cycle += 1
            total_cycles = "inf" if REPEAT_COUNT is None else str(REPEAT_COUNT)
            print(f"Sequence cycle {cycle}/{total_cycles}")

            for index, commands in enumerate(sequence, start=1):
                target = apply_step(base, hand, commands)
                print(f"Step {index}/{len(sequence)}: {describe_step(commands)}")
                hand.set_positions(target, clip=True)

                if WAIT_UNTIL_REACHED:
                    reached = wait_until_reached_with_error_print(hand, target, commands)
                    if not reached:
                        print("Warning: target was not reached before timeout.")

                if WAIT_BETWEEN_MOVEMENTS_S > 0:
                    time.sleep(WAIT_BETWEEN_MOVEMENTS_S)
                base = target

            if REPEAT_COUNT is None or cycle < REPEAT_COUNT:
                if REPEAT_DELAY_S > 0:
                    time.sleep(REPEAT_DELAY_S)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        hand.shutdown()


if __name__ == "__main__":
    main()
