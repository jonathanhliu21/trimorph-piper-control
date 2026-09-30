from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence


def load_hand_config(
    path: Path, port: str, active_motors: Sequence[int] | None = None
):
    from midas_hand_api import HandConfig

    if not path.is_file():
        raise FileNotFoundError(f"TRIMORPH hand config not found: {path}")
    config = HandConfig.load(path)
    if config.motor_model != "xl330-m077-t":
        raise RuntimeError(
            f"Expected motor_model 'xl330-m077-t' in {path}, got {config.motor_model!r}"
        )
    selected_motor_ids = (
        tuple(int(motor_id) for motor_id in active_motors)
        if active_motors is not None
        else tuple(config.motor_ids)
    )
    return _select_motors(config, selected_motor_ids, path=path, port=port)


def load_follower_config(
    path: Path, port: str, active_motors: Sequence[int] | None = None
):
    """Load an optional MIDAS-family follower without TRIMORPH-specific constraints."""
    from midas_hand_api import HandConfig

    if not path.is_file():
        raise FileNotFoundError(f"Follower hand config not found: {path}")
    config = HandConfig.load(path)
    selected_motor_ids = (
        tuple(int(motor_id) for motor_id in active_motors)
        if active_motors is not None
        else tuple(config.motor_ids)
    )
    return _select_motors(config, selected_motor_ids, path=path, port=port)


def _select_motors(config, motor_ids: Sequence[int], *, path: Path, port: str):
    """Return a config whose vectors and bus operations contain only selected IDs."""

    selected_motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
    if not selected_motor_ids:
        raise ValueError("At least one active motor ID is required")
    if len(set(selected_motor_ids)) != len(selected_motor_ids):
        raise ValueError(f"Active motor IDs must be unique: {selected_motor_ids}")

    configured_motor_ids = tuple(int(motor_id) for motor_id in config.motor_ids)
    missing = sorted(set(selected_motor_ids) - set(configured_motor_ids))
    if missing:
        raise RuntimeError(
            f"Active motor IDs {missing} are not present in {path}; "
            f"configured motor IDs are {configured_motor_ids}"
        )

    # This is the same selection mechanism used by two_joint_test.py.  It also
    # subsets calibration and limit vectors, rather than merely changing the
    # list of IDs passed to the hardware object.
    return replace(config.subset(selected_motor_ids), port=port)


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
