"""Fit q_M[2:4] = A q_T[2:4] + b from thumb correspondences."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Support both direct execution and package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from data_collection.collection.thumb_retargeting import (
    MODEL_FORMAT,
    THUMB_RETARGET_MOTOR_IDS,
)


DEFAULT_INPUT = Path("thumb_tip_matching.csv")
DEFAULT_OUTPUT = Path("thumb_retarget_model.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the linear TRIMORPH-to-MIDAS model for thumb joints 2 and 3."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_correspondences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    trimorph_fields = [
        f"trimorph_q{motor_id}" for motor_id in THUMB_RETARGET_MOTOR_IDS
    ]
    midas_fields = [f"midas_q{motor_id}" for motor_id in THUMB_RETARGET_MOTOR_IDS]
    trimorph_rows: list[list[float]] = []
    midas_rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [
            field
            for field in trimorph_fields + midas_fields
            if field not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(f"Calibration CSV is missing columns: {missing}")
        for line_number, row in enumerate(reader, start=2):
            try:
                trimorph_rows.append([float(row[field]) for field in trimorph_fields])
                midas_rows.append([float(row[field]) for field in midas_fields])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numeric calibration value on CSV line {line_number}"
                ) from exc
    trimorph = np.asarray(trimorph_rows, dtype=np.float64)
    midas = np.asarray(midas_rows, dtype=np.float64)
    minimum_samples = len(THUMB_RETARGET_MOTOR_IDS) + 1
    if len(trimorph) < minimum_samples:
        raise ValueError(
            f"At least {minimum_samples} point correspondences are required"
        )
    if not np.isfinite(trimorph).all() or not np.isfinite(midas).all():
        raise ValueError("Calibration CSV contains NaN or Inf")
    return trimorph, midas


def fit_linear_model(
    trimorph: np.ndarray, midas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Solve the multivariate affine least-squares problem with an SVD."""

    trimorph = np.asarray(trimorph, dtype=np.float64)
    midas = np.asarray(midas, dtype=np.float64)
    dimensions = len(THUMB_RETARGET_MOTOR_IDS)
    if trimorph.ndim != 2 or trimorph.shape[1] != dimensions:
        raise ValueError(f"TRIMORPH samples must have shape (N, {dimensions})")
    if midas.shape != trimorph.shape:
        raise ValueError(
            f"MIDAS samples must have the same (N, {dimensions}) shape"
        )
    design = np.column_stack((trimorph, np.ones(len(trimorph))))
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, midas, rcond=None
    )
    if rank < design.shape[1]:
        raise ValueError(
            "Calibration poses do not span both retargeted TRIMORPH thumb "
            f"dimensions (design rank {rank}, expected {design.shape[1]})"
        )
    matrix = coefficients[:dimensions, :].T
    bias = coefficients[dimensions, :]
    predicted = trimorph @ matrix.T + bias
    errors = predicted - midas
    rmse_per_joint = np.sqrt(np.mean(errors**2, axis=0))
    metrics: dict[str, object] = {
        "sample_count": int(len(trimorph)),
        "design_rank": int(rank),
        "singular_values": singular_values.tolist(),
        "rmse_per_joint_rad": rmse_per_joint.tolist(),
        "rmse_total_rad": float(np.sqrt(np.mean(errors**2))),
    }
    return matrix, bias, metrics


def run(input_path: Path, output_path: Path) -> int:
    trimorph, midas = load_correspondences(input_path)
    matrix, bias, metrics = fit_linear_model(trimorph, midas)
    payload = {
        "format": MODEL_FORMAT,
        "equation": "q_M[2:4] = A @ q_T[2:4] + b",
        "trimorph_motor_ids": list(THUMB_RETARGET_MOTOR_IDS),
        "midas_motor_ids": list(THUMB_RETARGET_MOTOR_IDS),
        "A": matrix.tolist(),
        "b": bias.tolist(),
        "fit": metrics,
        "source_csv": str(input_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(f"Saved TRIMORPH -> MIDAS thumb model to {output_path}")
    print(f"Samples: {metrics['sample_count']}; rank: {metrics['design_rank']}")
    print(
        "RMSE per MIDAS joint [rad]: "
        + np.array2string(
            np.asarray(metrics["rmse_per_joint_rad"]), precision=6
        )
    )
    print("A =")
    print(matrix)
    print("b =")
    print(bias)
    return 0


def main() -> int:
    args = parse_args()
    return run(args.input.expanduser(), args.output.expanduser())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
