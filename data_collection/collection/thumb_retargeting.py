from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


THUMB_MOTOR_IDS = (0, 1, 2, 3)
THUMB_RETARGET_MOTOR_IDS = (2, 3)
MODEL_FORMAT = "trimorph-thumb-linear-v2"


@dataclass(frozen=True)
class ThumbRetargetModel:
    """Linear map from TRIMORPH joints 2-3 to MIDAS joints 2-3."""

    matrix: np.ndarray
    bias: np.ndarray
    trimorph_motor_ids: tuple[int, ...] = THUMB_RETARGET_MOTOR_IDS
    midas_motor_ids: tuple[int, ...] = THUMB_RETARGET_MOTOR_IDS

    def __post_init__(self) -> None:
        if (
            self.trimorph_motor_ids != THUMB_RETARGET_MOTOR_IDS
            or self.midas_motor_ids != THUMB_RETARGET_MOTOR_IDS
        ):
            raise ValueError(
                "Thumb model must map TRIMORPH joints 2-3 to MIDAS joints 2-3"
            )
        matrix = np.asarray(self.matrix, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        expected = (len(self.midas_motor_ids), len(self.trimorph_motor_ids))
        if matrix.shape != expected:
            raise ValueError(
                f"Thumb model A has shape {matrix.shape}; expected {expected}"
            )
        if bias.shape != (len(self.midas_motor_ids),):
            raise ValueError(
                f"Thumb model b has shape {bias.shape}; expected "
                f"({len(self.midas_motor_ids)},)"
            )
        if not np.isfinite(matrix).all() or not np.isfinite(bias).all():
            raise ValueError("Thumb model contains NaN or Inf")
        if len(set(self.trimorph_motor_ids)) != len(self.trimorph_motor_ids):
            raise ValueError("TRIMORPH motor IDs in thumb model must be unique")
        if len(set(self.midas_motor_ids)) != len(self.midas_motor_ids):
            raise ValueError("MIDAS motor IDs in thumb model must be unique")
        object.__setattr__(self, "matrix", matrix.copy())
        object.__setattr__(self, "bias", bias.copy())

    def predict(self, trimorph_positions: Sequence[float]) -> np.ndarray:
        positions = np.asarray(trimorph_positions, dtype=np.float64)
        if positions.shape != (len(self.trimorph_motor_ids),):
            raise ValueError("TRIMORPH thumb position does not match model input size")
        if not np.isfinite(positions).all():
            raise ValueError("TRIMORPH thumb position contains NaN or Inf")
        prediction = self.matrix @ positions + self.bias
        if not np.isfinite(prediction).all():
            raise ValueError("Thumb model produced NaN or Inf")
        return prediction


def load_thumb_retarget_model(path: Path) -> ThumbRetargetModel:
    if not path.is_file():
        raise FileNotFoundError(
            f"Thumb retarget model not found: {path}. "
            "Run fit_model_to_endpoints.py first."
        )
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("format") != MODEL_FORMAT:
        raise ValueError(
            f"Unsupported thumb model format {payload.get('format')!r} in {path}; "
            "re-run fit_model_to_endpoints.py to create a joints 2-3 model"
        )
    model = ThumbRetargetModel(
        matrix=np.asarray(payload["A"], dtype=np.float64),
        bias=np.asarray(payload["b"], dtype=np.float64),
        trimorph_motor_ids=tuple(int(value) for value in payload["trimorph_motor_ids"]),
        midas_motor_ids=tuple(int(value) for value in payload["midas_motor_ids"]),
    )
    return model


def apply_thumb_retargeting(
    positions: Sequence[float],
    motor_ids: Sequence[int],
    model: ThumbRetargetModel,
) -> np.ndarray:
    """Replace MIDAS thumb targets in a full follower-order target vector."""

    result = np.asarray(positions, dtype=np.float64).copy()
    ids = tuple(int(motor_id) for motor_id in motor_ids)
    if result.shape != (len(ids),):
        raise ValueError("Follower target and motor IDs must have matching lengths")
    index_by_id = {motor_id: index for index, motor_id in enumerate(ids)}
    required = set(model.trimorph_motor_ids) | set(model.midas_motor_ids)
    present = required & set(ids)
    if present and present != required:
        missing = sorted(required - set(ids))
        raise ValueError(
            "Thumb retargeting requires all model motor IDs; "
            f"missing active motor IDs {missing}"
        )
    if not present:
        return result

    trimorph_thumb = np.asarray(
        [result[index_by_id[motor_id]] for motor_id in model.trimorph_motor_ids],
        dtype=np.float64,
    )
    prediction = model.predict(trimorph_thumb)
    for motor_id, value in zip(model.midas_motor_ids, prediction):
        result[index_by_id[motor_id]] = value
    return result


def build_follower_target(
    leader_positions: Sequence[float],
    motor_ids: Sequence[int],
    *,
    command_scale: float,
    joint_command_scales: Sequence[float],
    thumb_model: ThumbRetargetModel | None,
    joint_command_offsets: Sequence[float] | None = None,
) -> np.ndarray:
    """Apply an affine joint map, then replace modeled MIDAS joints 2-3.

    Each offset is the MIDAS target when the corresponding TRIMORPH joint is
    zero: ``midas = trimorph * global_scale * joint_scale + joint_offset``.
    The learned thumb model owns its own bias, so it replaces both the generic
    scale and offset for its output joints.
    """

    leader = np.asarray(leader_positions, dtype=np.float64)
    scales = np.asarray(joint_command_scales, dtype=np.float64)
    expected_shape = (len(motor_ids),)
    if leader.shape != expected_shape or scales.shape != expected_shape:
        raise ValueError("Follower target inputs must match the motor ID count")
    offsets = (
        np.zeros(expected_shape, dtype=np.float64)
        if joint_command_offsets is None
        else np.asarray(joint_command_offsets, dtype=np.float64)
    )
    if offsets.shape != expected_shape:
        raise ValueError("Follower command offsets must match the motor ID count")
    if not np.isfinite(offsets).all():
        raise ValueError("Follower command offsets must be finite")
    target = leader * float(command_scale) * scales + offsets
    if thumb_model is not None:
        retargeted = apply_thumb_retargeting(leader, motor_ids, thumb_model)
        thumb_ids = set(thumb_model.midas_motor_ids)
        for index, motor_id in enumerate(motor_ids):
            if int(motor_id) in thumb_ids:
                target[index] = retargeted[index]
    return target
