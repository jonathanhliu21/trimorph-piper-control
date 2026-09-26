from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import TRIMORPH_MOTOR_IDS


def load_hand_config(path: Path, port: str):
    from midas_hand_api import HandConfig

    if not path.is_file():
        raise FileNotFoundError(f"TRIMORPH hand config not found: {path}")
    config = HandConfig.load(path)
    config = replace(config, port=port)
    if config.motor_model != "xl330-m077-t":
        raise RuntimeError(
            f"Expected motor_model 'xl330-m077-t' in {path}, got {config.motor_model!r}"
        )
    if tuple(config.motor_ids) != TRIMORPH_MOTOR_IDS:
        raise RuntimeError(
            f"Expected motor IDs {TRIMORPH_MOTOR_IDS} in {path}, got {config.motor_ids}"
        )
    return config


def load_follower_config(path: Path, port: str):
    """Load an optional MIDAS-family follower without TRIMORPH-specific constraints."""
    from midas_hand_api import HandConfig

    if not path.is_file():
        raise FileNotFoundError(f"Follower hand config not found: {path}")
    return replace(HandConfig.load(path), port=port)


def verify_connected_hand(hand, *, role: str) -> None:
    responses = hand.ping()
    missing = sorted(set(hand.motor_ids) - set(responses))
    if missing:
        raise RuntimeError(f"{role.capitalize()} hand is missing motor IDs: {missing}")
    wrong_models = {
        motor_id: model_number
        for motor_id, model_number in responses.items()
        if model_number != hand.config.expected_model_number
    }
    if wrong_models:
        raise RuntimeError(
            f"{role.capitalize()} hand has unexpected motor models: {wrong_models}; "
            f"expected {hand.config.expected_model_number}"
        )
