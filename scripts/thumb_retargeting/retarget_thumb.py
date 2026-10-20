"""Interactively collect corresponding TRIMORPH and MIDAS thumb poses."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Support both direct execution and package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from data_collection.collection.follower_hand_worker import prepare_follower_hand
from data_collection.collection.hand_common import (
    load_follower_config,
    load_hand_config,
    verify_connected_hand,
)
from data_collection.collection.thumb_retargeting import THUMB_MOTOR_IDS, THUMB_RETARGET_MOTOR_IDS


DEFAULT_OUTPUT = Path("thumb_tip_matching.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move the MIDAS thumb through calibration poses and record the "
            "corresponding operator-held TRIMORPH thumb poses."
        )
    )
    parser.add_argument("--trimorph-config", type=Path, required=True)
    parser.add_argument("--trimorph-port", required=True)
    parser.add_argument(
        "--midas-config",
        type=Path,
        default=Path("config/midas_hand_config.yaml"),
    )
    parser.add_argument("--midas-port", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-points", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--limit-margin",
        type=float,
        default=0.05,
        help="Fraction of each MIDAS joint range excluded at both limits.",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=0.04,
        help="Maximum commanded change per interpolation step in radians.",
    )
    parser.add_argument("--command-rate", type=float, default=50.0)
    parser.add_argument("--settle-time", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument(
        "--no-configure-midas",
        action="store_true",
        help="Do not rewrite MIDAS operating-mode and gain registers.",
    )
    return parser.parse_args()


def generate_waypoints(
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    count: int = 40,
    *,
    seed: int = 7,
    margin: float = 0.05,
) -> np.ndarray:
    """Generate a deterministic Latin-hypercube sample of the joint box."""

    lower = np.asarray(lower_limits, dtype=np.float64)
    upper = np.asarray(upper_limits, dtype=np.float64)
    dimensions = len(THUMB_RETARGET_MOTOR_IDS)
    if lower.shape != (dimensions,) or upper.shape != (dimensions,):
        raise ValueError(
            f"Retargeted thumb limits must each contain {dimensions} values"
        )
    if count < dimensions + 1:
        raise ValueError(
            f"At least {dimensions + 1} calibration points are required"
        )
    if not 0.0 <= margin < 0.5:
        raise ValueError("--limit-margin must be in [0, 0.5)")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Thumb limits must be finite")
    if np.any(upper <= lower):
        raise ValueError("Each thumb upper limit must exceed its lower limit")

    safe_lower = lower + margin * (upper - lower)
    safe_upper = upper - margin * (upper - lower)
    rng = np.random.default_rng(seed)
    unit = np.empty((count, dimensions), dtype=np.float64)
    for joint in range(dimensions):
        unit[:, joint] = (rng.permutation(count) + 0.5) / count
    return safe_lower + unit * (safe_upper - safe_lower)


def generate_midas_waypoints(
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    count: int = 40,
    *,
    seed: int = 7,
    margin: float = 0.05,
) -> np.ndarray:
    """Generate four-joint commands with joints 0-1 fixed at zero."""

    lower = np.asarray(lower_limits, dtype=np.float64)
    upper = np.asarray(upper_limits, dtype=np.float64)
    if lower.shape != (4,) or upper.shape != (4,):
        raise ValueError("MIDAS thumb limits must each contain four values")
    retarget_waypoints = generate_waypoints(
        lower[2:4], upper[2:4], count, seed=seed, margin=margin
    )
    return np.column_stack(
        (np.zeros((len(retarget_waypoints), 2)), retarget_waypoints)
    )


def order_waypoints(waypoints: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Greedily shorten travel between samples, normalized by joint range."""

    points = np.asarray(waypoints, dtype=np.float64)
    current = np.asarray(start, dtype=np.float64)
    scale = np.ptp(points, axis=0)
    scale[scale == 0.0] = 1.0
    remaining = list(range(len(points)))
    ordered: list[np.ndarray] = []
    while remaining:
        index = min(
            remaining,
            key=lambda candidate: float(
                np.linalg.norm((points[candidate] - current) / scale)
            ),
        )
        ordered.append(points[index])
        current = points[index]
        remaining.remove(index)
    return np.asarray(ordered)


def move_safely(hand, start, target, *, max_step: float, rate_hz: float) -> None:
    start_array = np.asarray(start, dtype=np.float64)
    target_array = hand.clip_positions(np.asarray(target, dtype=np.float64))
    largest_delta = float(np.max(np.abs(target_array - start_array)))
    steps = max(1, int(np.ceil(largest_delta / max_step)))
    period = 1.0 / rate_hz
    for step in range(1, steps + 1):
        command = start_array + (target_array - start_array) * (step / steps)
        hand.set_positions(command, clip=True)
        time.sleep(period)


def average_positions(trimorph, midas, *, samples: int, rate_hz: float):
    trimorph_samples: list[np.ndarray] = []
    midas_samples: list[np.ndarray] = []
    period = 1.0 / rate_hz
    for sample_index in range(samples):
        trimorph_samples.append(np.asarray(trimorph.read_pos(), dtype=np.float64))
        midas_samples.append(np.asarray(midas.read_pos(), dtype=np.float64))
        if sample_index + 1 < samples:
            time.sleep(period)
    trimorph_mean = np.mean(trimorph_samples, axis=0)
    midas_mean = np.mean(midas_samples, axis=0)
    if not np.isfinite(trimorph_mean).all() or not np.isfinite(midas_mean).all():
        raise RuntimeError("Hand measurement contained NaN or Inf")
    return trimorph_mean, midas_mean


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["point_index"]
    fieldnames += [f"trimorph_q{motor_id}" for motor_id in THUMB_MOTOR_IDS]
    fieldnames += [f"midas_q{motor_id}" for motor_id in THUMB_MOTOR_IDS]
    fieldnames += [f"midas_command_q{motor_id}" for motor_id in THUMB_MOTOR_IDS]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_args(args: argparse.Namespace) -> None:
    if args.trimorph_port == args.midas_port:
        raise ValueError("TRIMORPH and MIDAS ports must be different")
    if args.num_points < len(THUMB_RETARGET_MOTOR_IDS) + 1:
        raise ValueError("--num-points must be at least 3")
    if args.max_step <= 0.0 or args.command_rate <= 0.0:
        raise ValueError("--max-step and --command-rate must be positive")
    if args.settle_time < 0.0:
        raise ValueError("--settle-time must be non-negative")
    if args.samples <= 0 or args.sample_rate <= 0.0:
        raise ValueError("--samples and --sample-rate must be positive")


def run(args: argparse.Namespace) -> int:
    from midas_hand_api import MidasHand

    validate_args(args)
    trimorph_config = load_hand_config(
        args.trimorph_config.expanduser(), args.trimorph_port, THUMB_MOTOR_IDS
    )
    midas_config = load_follower_config(
        args.midas_config.expanduser(), args.midas_port, THUMB_MOTOR_IDS
    )
    trimorph = MidasHand(config=trimorph_config, autoconnect=True)
    midas = None
    rows: list[dict[str, object]] = []
    output = args.output.expanduser()
    try:
        verify_connected_hand(trimorph, role="TRIMORPH")
        trimorph.disable_torque()
        midas = MidasHand(config=midas_config, autoconnect=True)
        verify_connected_hand(midas, role="MIDAS")
        prepare_follower_hand(midas, configure_hardware=not args.no_configure_midas)

        current = midas.clip_positions(midas.read_pos())
        midas.set_positions(current, clip=True)
        midas.enable_torque()
        waypoints = generate_midas_waypoints(
            np.asarray(midas.config.joint_lower_limits_array, dtype=np.float64),
            np.asarray(midas.config.joint_upper_limits_array, dtype=np.float64),
            args.num_points,
            seed=args.seed,
            margin=args.limit_margin,
        )
        waypoints = order_waypoints(waypoints, current)
        print(
            f"Connected both thumbs. Collecting {len(waypoints)} points. "
            "TRIMORPH torque is disabled; MIDAS torque is enabled. "
            "MIDAS joints 0 and 1 remain at zero."
        )
        for point_index, command in enumerate(waypoints):
            move_safely(
                midas,
                current,
                command,
                max_step=args.max_step,
                rate_hz=args.command_rate,
            )
            current = command.copy()
            time.sleep(args.settle_time)
            print(
                f"\n[{point_index + 1}/{len(waypoints)}] MIDAS target: "
                + np.array2string(command, precision=3, suppress_small=True)
            )
            response = input(
                "Hold the TRIMORPH thumb at the matching pose, then press Enter "
                "(or type q to finish early): "
            )
            if response.strip().lower() in {"q", "quit"}:
                break
            trimorph_position, midas_position = average_positions(
                trimorph,
                midas,
                samples=args.samples,
                rate_hz=args.sample_rate,
            )
            row: dict[str, object] = {"point_index": point_index}
            for motor_id, value in zip(THUMB_MOTOR_IDS, trimorph_position):
                row[f"trimorph_q{motor_id}"] = f"{float(value):.9f}"
            for motor_id, value in zip(THUMB_MOTOR_IDS, midas_position):
                row[f"midas_q{motor_id}"] = f"{float(value):.9f}"
            for motor_id, value in zip(THUMB_MOTOR_IDS, command):
                row[f"midas_command_q{motor_id}"] = f"{float(value):.9f}"
            rows.append(row)
            save_rows(output, rows)
            print(f"Captured point {point_index + 1}; checkpoint saved to {output}")
    except KeyboardInterrupt:
        print("\nCalibration interrupted; saving completed points.")
    finally:
        if rows:
            save_rows(output, rows)
        try:
            if midas is not None:
                try:
                    midas.disable_torque()
                finally:
                    midas.shutdown()
        finally:
            try:
                trimorph.disable_torque()
            finally:
                trimorph.shutdown()

    if not rows:
        print("No calibration points were captured.")
        return 1
    print(f"Saved {len(rows)} point correspondences to {output}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
