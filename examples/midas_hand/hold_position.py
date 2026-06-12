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
#   python3 examples/midas_hand/hold_position.py
# ---------------------------------------------------------------------------

# Target position per finger. Each command is ["finger", values...].
# Fingers use 3 values. Thumb uses 4 values.
# Motors not listed here are commanded to zero.
TARGET = [
    ["thumb",  -0.81147584, -0.1196505,  -0.50467967,  0.02607767],
    ["index",  -1.38279631, -1.5146604,   0.1318642],
    ["middle", -1.18951473, -1.72328179,  0.11118459],
    ["ring",   -1.34751475, -1.59902935,  0.04761155],
]

# Connection/config. Leave PORT as None to auto-detect.
BAUDRATE = 4_000_000
CONFIG_PATH = DEFAULT_CONFIG_PATH
OPERATING_MODE_POSITION = 3

# Motion behavior.
POSITION_P_GAIN = 800
POSITION_I_GAIN = 0
POSITION_D_GAIN = 400
GOAL_CURRENT_LIMIT_MA = None
PROFILE_VELOCITY_RAD_S = 0.4
PROFILE_ACCELERATION_RAD_S_2 = 1.0


class HandArgs:
    port = None
    baudrate = BAUDRATE
    config = str(CONFIG_PATH)


def connect_position_mode_hand(enable_torque=True):
    _, _, MidasHand = import_hand_api()
    config = replace(load_hand_config(HandArgs), operating_mode=OPERATING_MODE_POSITION)
    hand = MidasHand(config)
    hand.configure(enable_torque=enable_torque)
    return hand


def build_target(hand):
    target = np.zeros(len(hand.motor_ids), dtype=np.float64)
    for command in TARGET:
        finger = command[0]
        if finger not in FINGER_IDS:
            raise ValueError(f"Unknown finger {finger!r}. Choose one of {sorted(FINGER_IDS)}")
        values = np.asarray(command[1:], dtype=np.float64)
        expected_dim = len(FINGER_IDS[finger])
        if len(values) != expected_dim:
            raise ValueError(f"{finger} expects {expected_dim} values, got {len(values)}")
        target = apply_finger_values(target, hand, finger, values)
    return target


def main():
    hand = connect_position_mode_hand(enable_torque=True)
    try:
        print(f"Set operating mode: Position Control ({OPERATING_MODE_POSITION})")
        hand.set_gains(p=POSITION_P_GAIN, i=POSITION_I_GAIN, d=POSITION_D_GAIN)
        print(f"Set position gains: P={POSITION_P_GAIN}, I={POSITION_I_GAIN}, D={POSITION_D_GAIN}")
        if GOAL_CURRENT_LIMIT_MA is not None:
            hand.set_current_limit(GOAL_CURRENT_LIMIT_MA)
            print(f"Set goal current limit: {GOAL_CURRENT_LIMIT_MA} mA")
        if PROFILE_VELOCITY_RAD_S is not None or PROFILE_ACCELERATION_RAD_S_2 is not None:
            hand.set_motion_profile(
                profile_velocity_rad_s=PROFILE_VELOCITY_RAD_S,
                profile_acceleration_rad_s2=PROFILE_ACCELERATION_RAD_S_2,
            )

        print(f"Hand connected on {hand.port}")
        target = build_target(hand)
        print(f"Moving to target: {np.round(target, 4).tolist()}")
        hand.set_positions(target, clip=True)
        print("Holding position. Press Ctrl+C to exit.")
        while True:
            cur = hand.read_cur()
            print(f"Current (mA): {np.round(cur, 1).tolist()}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        hand.shutdown()


if __name__ == "__main__":
    main()
