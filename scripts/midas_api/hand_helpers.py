from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


MIDAS_HAND_API_CHECKOUT = Path(__file__).resolve().parents[3] / "midas_hand_api"
if MIDAS_HAND_API_CHECKOUT.exists() and str(MIDAS_HAND_API_CHECKOUT) not in sys.path:
    sys.path.insert(0, str(MIDAS_HAND_API_CHECKOUT))

DEFAULT_CONFIG_PATH = Path.home() / ".midas_hand" / "config.yaml"

THUMB_IDS = (0, 1, 2, 3)
FINGER_IDS = {
    "thumb": THUMB_IDS,
    "index": (4, 5, 6),
    "pointer": (4, 5, 6),
    "middle": (7, 8, 9),
    "ring": (10, 11, 12),
}


def import_hand_api():
    from midas_hand_api import DEFAULT_CONFIG_PATH as api_default_config_path
    from midas_hand_api import HandConfig, MidasHand

    return api_default_config_path, HandConfig, MidasHand


def add_hand_connection_args(parser) -> None:
    parser.add_argument("--port", default=None, help="Dynamixel hand serial port.")
    parser.add_argument("--baudrate", type=int, default=4_000_000)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Hand calibration YAML. Default: {DEFAULT_CONFIG_PATH}",
    )


def load_hand_config(args):
    _, HandConfig, _ = import_hand_api()
    config_path = Path(args.config).expanduser()
    if config_path.exists():
        config = HandConfig.load(config_path)
    else:
        config = HandConfig.xm335_t323()

    updates = {}
    if args.port:
        updates["port"] = args.port
    if args.baudrate:
        updates["baudrate"] = args.baudrate
    if updates:
        config = replace(config, **updates)
    return config


def connect_hand(args, *, enable_torque: bool):
    _, _, MidasHand = import_hand_api()
    hand = MidasHand(load_hand_config(args))
    hand.configure(enable_torque=enable_torque)
    return hand


def parse_array(text: str) -> np.ndarray:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Expected JSON array syntax, for example '[0.1, 0.2, 0.3]' "
            "or '[[0, 0, 0], [0.2, 0.3, 0.1]]'"
        ) from exc

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    raise ValueError("Expected a 1D vector or 2D sequence array")


def apply_finger_values(base_positions, hand, finger, values):
    motor_ids = FINGER_IDS[finger]
    target = np.asarray(base_positions, dtype=np.float64).copy()
    for motor_id, value in zip(motor_ids, values):
        target[hand.motor_ids.index(motor_id)] = value
    return target


def wait_until_reached_or_timeout(
    hand,
    target,
    *,
    timeout_s: float,
    tolerance_rad: float,
    poll_s: float = 0.02,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current = hand.read_pos()
        if np.all(np.abs(current - target) <= tolerance_rad):
            return True
        time.sleep(poll_s)
    return False
