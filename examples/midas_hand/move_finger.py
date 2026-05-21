import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.midas_api import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    FINGER_IDS,
    apply_finger_values,
    connect_hand,
    wait_until_reached_or_timeout,
)


# ---------------------------------------------------------------------------
# Edit these values, then run:
#   python3 examples/midas_hand/move_finger.py
# ---------------------------------------------------------------------------

# Options: "thumb", "index", "pointer", "middle", "ring"
FINGER = "index"

# Fingers use 3 values. Thumb uses 4 values.
# A single row moves once; multiple rows run as a sequence.
SEQUENCE = [
    [0.0, 0.0, 0.0],
    [0.1, 0.2, 0.3],
]

# Seconds to wait after each commanded movement.
WAIT_BETWEEN_MOVEMENTS_S = 0.5

# Use "current" to keep non-commanded motors where they are, or "zero" to
# command non-commanded motors to zero while moving this finger.
BASE_POSE = "current"

# Connection/config. Leave PORT as None to auto-detect.
PORT = None
BAUDRATE = 1_000_000
CONFIG_PATH = DEFAULT_CONFIG_PATH

# Motion behavior.
WAIT_UNTIL_REACHED = False
REACH_TIMEOUT_S = 5.0
REACH_TOLERANCE_RAD = 0.05
PROFILE_VELOCITY_RAD_S = None
PROFILE_ACCELERATION_RAW = None


class HandArgs:
    port = PORT
    baudrate = BAUDRATE
    config = str(CONFIG_PATH)


def validate_sequence(sequence, finger):
    if finger not in FINGER_IDS:
        raise ValueError(f"Unknown finger {finger!r}. Choose one of {sorted(FINGER_IDS)}")

    array = np.asarray(sequence, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("SEQUENCE must be a vector or a 2D sequence array")

    expected_dim = len(FINGER_IDS[finger])
    if array.shape[1] != expected_dim:
        raise ValueError(f"{finger} expects {expected_dim} values, got {array.shape[1]}")
    return array


def main():
    if WAIT_BETWEEN_MOVEMENTS_S < 0:
        raise ValueError("WAIT_BETWEEN_MOVEMENTS_S must be 0 or greater")
    if BASE_POSE not in ("current", "zero"):
        raise ValueError("BASE_POSE must be 'current' or 'zero'")
    if REACH_TIMEOUT_S < 0:
        raise ValueError("REACH_TIMEOUT_S must be 0 or greater")
    if REACH_TOLERANCE_RAD < 0:
        raise ValueError("REACH_TOLERANCE_RAD must be 0 or greater")

    sequence = validate_sequence(SEQUENCE, FINGER)

    hand = connect_hand(HandArgs, enable_torque=True)
    try:
        if PROFILE_VELOCITY_RAD_S is not None or PROFILE_ACCELERATION_RAW is not None:
            hand.set_motion_profile(
                profile_velocity_rad_s=PROFILE_VELOCITY_RAD_S,
                profile_acceleration_raw=PROFILE_ACCELERATION_RAW,
            )

        print(f"Hand connected on {hand.port}")
        base = np.zeros(len(hand.motor_ids)) if BASE_POSE == "zero" else hand.read_pos()

        for index, values in enumerate(sequence, start=1):
            target = apply_finger_values(base, hand, FINGER, values)
            print(f"{FINGER} target {index}/{len(sequence)}: {values.tolist()}")
            hand.set_positions(target, clip=True)

            if WAIT_UNTIL_REACHED:
                reached = wait_until_reached_or_timeout(
                    hand,
                    target,
                    timeout_s=REACH_TIMEOUT_S,
                    tolerance_rad=REACH_TOLERANCE_RAD,
                )
                if not reached:
                    print("Warning: target was not reached before timeout.")

            if WAIT_BETWEEN_MOVEMENTS_S > 0:
                time.sleep(WAIT_BETWEEN_MOVEMENTS_S)
            base = target

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        hand.shutdown()


if __name__ == "__main__":
    main()
