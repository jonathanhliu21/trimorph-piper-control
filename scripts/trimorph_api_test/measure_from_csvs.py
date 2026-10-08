#!/usr/bin/env python3
"""Compute absolute position-error statistics from telemetry CSV files.

The input format is the one written by ``two_joint_test.py`` and contains
paired columns named::

    motor_<id>_target_position_rad
    motor_<id>_actual_position_rad

For each motor, position error is ``target - actual``. The reported metrics are
mean absolute error (MAE) and several percentiles of ``abs(position error)``
over every row in the file.

Example:

    python scripts/trimorph_api_test/measure_from_csvs.py \
        output/contact_test/medium_open_close.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


TARGET_COLUMN = re.compile(r"^motor_(\d+)_target_position_rad$")
ABSOLUTE_ERROR_PERCENTILES = (50, 90, 95, 99, 100)


@dataclass(frozen=True)
class MotorPositionError:
    motor_id: int
    sample_count: int
    mae_rad: float
    percentile_errors_rad: tuple[float, ...]
    _absolute_errors_rad: tuple[float, ...]


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile, matching NumPy's default."""

    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower_index = math.floor(index)
    upper_index = math.ceil(index)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = index - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def _percentile_errors(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        _percentile(values, percentile)
        for percentile in ABSOLUTE_ERROR_PERCENTILES
    )


def measure_position_mae(path: Path) -> list[MotorPositionError]:
    """Return per-motor absolute position-error statistics for one CSV."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")

        motor_ids = sorted(
            int(match.group(1))
            for field in reader.fieldnames
            if (match := TARGET_COLUMN.fullmatch(field))
        )
        if not motor_ids:
            raise ValueError(f"{path}: no motor target-position columns found")

        actual_columns = {
            motor_id: f"motor_{motor_id}_actual_position_rad"
            for motor_id in motor_ids
        }
        missing = [
            column
            for column in actual_columns.values()
            if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")

        absolute_errors = {motor_id: [] for motor_id in motor_ids}
        for row_number, row in enumerate(reader, start=2):
            for motor_id in motor_ids:
                target_column = f"motor_{motor_id}_target_position_rad"
                actual_column = actual_columns[motor_id]
                try:
                    target = float(row[target_column])
                    actual = float(row[actual_column])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}:{row_number}: non-numeric position for motor "
                        f"{motor_id}"
                    ) from exc
                if not math.isfinite(target) or not math.isfinite(actual):
                    raise ValueError(
                        f"{path}:{row_number}: non-finite position for motor "
                        f"{motor_id}"
                    )
                absolute_errors[motor_id].append(abs(target - actual))

    if not any(absolute_errors.values()):
        raise ValueError(f"{path}: contains no telemetry samples")

    return [
        MotorPositionError(
            motor_id=motor_id,
            sample_count=len(absolute_errors[motor_id]),
            mae_rad=sum(absolute_errors[motor_id]) / len(absolute_errors[motor_id]),
            percentile_errors_rad=_percentile_errors(absolute_errors[motor_id]),
            _absolute_errors_rad=tuple(absolute_errors[motor_id]),
        )
        for motor_id in motor_ids
    ]


def _format_percentiles(values: Sequence[float]) -> str:
    return ", ".join(
        f"p{percentile}={value:.6f}"
        for percentile, value in zip(ABSOLUTE_ERROR_PERCENTILES, values)
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_paths",
        type=Path,
        nargs="+",
        metavar="CSV",
        help="One or more telemetry CSV files produced by two_joint_test.py",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        for file_index, path in enumerate(args.csv_paths):
            results = measure_position_mae(path)
            if file_index:
                print()
            print(path)
            for result in results:
                print(
                    f"  motor {result.motor_id}: {result.mae_rad:.6f} rad "
                    f"({math.degrees(result.mae_rad):.3f} deg, "
                    f"n={result.sample_count})"
                )
                print(
                    "    absolute-error percentiles (rad): "
                    f"{_format_percentiles(result.percentile_errors_rad)}"
                )

            total_samples = sum(result.sample_count for result in results)
            overall_mae = sum(
                result.mae_rad * result.sample_count for result in results
            ) / total_samples
            print(
                f"  overall: {overall_mae:.6f} rad "
                f"({math.degrees(overall_mae):.3f} deg, n={total_samples})"
            )
            overall_absolute_errors = [
                error
                for result in results
                for error in result._absolute_errors_rad
            ]
            print(
                "    absolute-error percentiles (rad): "
                f"{_format_percentiles(_percentile_errors(overall_absolute_errors))}"
            )
    except (OSError, ValueError) as exc:
        _build_parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
