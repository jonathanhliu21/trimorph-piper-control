#!/usr/bin/env python3
"""Collect and characterize free-motion TRIMORPH -> MIDAS tracking error.

The leader is always torque-disabled.  The follower uses the same clipping,
interpolation, maximum-step limiting, and motion profile as
``collect_hand_only.py``.  No current/force feedback is read or applied.

Raw error is deliberately signed::

    raw_error_rad = q_issued_aligned_rad - q_robot_rad

MIDAS does not expose a device-side position timestamp through the API used by
this repository.  A measurement is therefore timestamped at the midpoint of
its host-side ``read_pos()`` call and aligned to the newest completed
``set_positions()`` command at or before that midpoint.  Command history and
read-call timing are retained in the output for later re-alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from collection.config import (
    TRIMORPH_MOTOR_IDS,
    joint_command_scales_for_motor_ids,
)
from collection.follower_hand_worker import prepare_follower_hand
from collection.hand_common import (
    load_follower_config,
    load_hand_config,
    verify_connected_hand,
)


# User-facing experiment defaults.  All are also exposed as command-line flags.
DEFAULT_DURATION_S = 30.0  # Use 0 to record until Ctrl-C.
DEFAULT_OUTPUT_ROOT = Path("data_collection/output/tracking_error")
DEFAULT_LEADER_RATE_HZ = 100.0
DEFAULT_FOLLOWER_MEASURE_RATE_HZ = 200.0
DEFAULT_FOLLOWER_COMMAND_RATE_HZ = 50.0
DEFAULT_SPEED_BIN_EDGES_RAD_S = (0.10, 0.50, 1.00)
DEFAULT_NEAR_ZERO_VELOCITY_RAD_S = 0.05
DEFAULT_REVERSAL_EXCLUSION_WINDOW_S = 0.15
DEFAULT_MAX_COMMAND_AGE_S = 0.10
DEFAULT_MAX_LEADER_AGE_S = 0.10
DEFAULT_FOLLOWER_COMMAND_SCALE = 1.0
DEFAULT_FOLLOWER_MAX_STEP_RAD = 0.15
DEFAULT_FOLLOWER_INTERPOLATION_ALPHA = 0.4

# Match collect_hand_only.py rather than the more general collection defaults.
DEFAULT_JOINT_COMMAND_SCALES = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 0.20,
    5: 0.5,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
    10: 1.0,
    11: 1.0,
    12: 1.0,
}

# Calibrated coordinates in this repository increase toward the open pose.
# Override individual motors with --opening-sign ID=-1 if a mechanism differs.
DEFAULT_OPENING_SIGNS = {motor_id: 1 for motor_id in TRIMORPH_MOTOR_IDS}

JOINT_NAMES = {
    0: "thumb_ip",
    1: "thumb_mcp",
    2: "thumb_cmc_side",
    3: "thumb_cmc_roll",
    4: "index_pip",
    5: "index_mcp_flex",
    6: "index_mcp_abduct",
    7: "middle_pip",
    8: "middle_mcp_flex",
    9: "middle_mcp_abduct",
    10: "ring_pip",
    11: "ring_mcp_flex",
    12: "ring_mcp_abduct",
}


@dataclass(frozen=True)
class TrackingConfig:
    leader_config: Path
    leader_port: str
    follower_config: Path
    follower_port: str
    output_root: Path = DEFAULT_OUTPUT_ROOT
    duration_s: float = DEFAULT_DURATION_S
    active_motors: tuple[int, ...] = TRIMORPH_MOTOR_IDS
    leader_rate_hz: float = DEFAULT_LEADER_RATE_HZ
    follower_measure_rate_hz: float = DEFAULT_FOLLOWER_MEASURE_RATE_HZ
    follower_command_rate_hz: float = DEFAULT_FOLLOWER_COMMAND_RATE_HZ
    follower_configure_hardware: bool = True
    follower_command_scale: float = DEFAULT_FOLLOWER_COMMAND_SCALE
    follower_max_step_rad: float = DEFAULT_FOLLOWER_MAX_STEP_RAD
    follower_interpolation_alpha: float = DEFAULT_FOLLOWER_INTERPOLATION_ALPHA
    joint_command_scales: dict[int, float] = field(
        default_factory=lambda: dict(DEFAULT_JOINT_COMMAND_SCALES)
    )
    opening_signs: dict[int, int] = field(
        default_factory=lambda: dict(DEFAULT_OPENING_SIGNS)
    )
    speed_bin_edges_rad_s: tuple[float, ...] = DEFAULT_SPEED_BIN_EDGES_RAD_S
    near_zero_velocity_rad_s: float = DEFAULT_NEAR_ZERO_VELOCITY_RAD_S
    reversal_exclusion_window_s: float = DEFAULT_REVERSAL_EXCLUSION_WINDOW_S
    max_command_age_s: float = DEFAULT_MAX_COMMAND_AGE_S
    max_leader_age_s: float = DEFAULT_MAX_LEADER_AGE_S
    make_plots: bool = True


@dataclass(frozen=True)
class LeaderSample:
    sequence: int
    monotonic_ns: int
    positions: np.ndarray


@dataclass(frozen=True)
class IssuedCommand:
    sequence: int
    call_start_ns: int
    issued_ns: int
    target: np.ndarray
    velocity_rad_s: np.ndarray
    leader_sample: LeaderSample | None


class SharedCapture:
    def __init__(self) -> None:
        self.shutdown = threading.Event()
        self.recording = threading.Event()
        self.recording_start_ns: int | None = None
        self._leader_lock = threading.Lock()
        self._leader: LeaderSample | None = None
        self.leader_motor_ids: tuple[int, ...] | None = None
        self.raw_rows: list[dict[str, object]] = []
        self.command_rows: list[dict[str, object]] = []

    def set_leader(self, sample: LeaderSample) -> None:
        with self._leader_lock:
            self._leader = LeaderSample(
                sample.sequence, sample.monotonic_ns, sample.positions.copy()
            )

    def latest_leader(self) -> LeaderSample | None:
        with self._leader_lock:
            sample = self._leader
            if sample is None:
                return None
            return LeaderSample(sample.sequence, sample.monotonic_ns, sample.positions.copy())


def _parse_key_values(values: Sequence[str], *, value_type, name: str) -> dict[int, object]:
    result: dict[int, object] = {}
    for value in values:
        try:
            key_text, setting_text = value.split("=", 1)
            key = int(key_text)
            setting = value_type(setting_text)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"{name} values must have the form MOTOR_ID=VALUE; got {value!r}"
            ) from exc
        result[key] = setting
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-config", type=Path, default=Path("config/hand.yaml"))
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--follower-config", type=Path, required=True)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_S,
        help="Recording seconds; 0 records until Ctrl-C (default: %(default)s).",
    )
    parser.add_argument(
        "--active-motors", nargs="+", type=int, default=list(TRIMORPH_MOTOR_IDS)
    )
    parser.add_argument("--leader-rate", type=float, default=DEFAULT_LEADER_RATE_HZ)
    parser.add_argument(
        "--measure-rate", type=float, default=DEFAULT_FOLLOWER_MEASURE_RATE_HZ
    )
    parser.add_argument(
        "--command-rate", type=float, default=DEFAULT_FOLLOWER_COMMAND_RATE_HZ
    )
    parser.add_argument(
        "--speed-bins", nargs="+", type=float,
        default=list(DEFAULT_SPEED_BIN_EDGES_RAD_S), metavar="RAD_S",
        help="Ascending upper edges; an additional final bin extends to infinity.",
    )
    parser.add_argument(
        "--near-zero-velocity", type=float,
        default=DEFAULT_NEAR_ZERO_VELOCITY_RAD_S,
    )
    parser.add_argument(
        "--reversal-window", type=float,
        default=DEFAULT_REVERSAL_EXCLUSION_WINDOW_S,
    )
    parser.add_argument("--max-command-age", type=float, default=DEFAULT_MAX_COMMAND_AGE_S)
    parser.add_argument("--max-leader-age", type=float, default=DEFAULT_MAX_LEADER_AGE_S)
    parser.add_argument(
        "--command-scale", type=float, default=DEFAULT_FOLLOWER_COMMAND_SCALE
    )
    parser.add_argument(
        "--max-step", type=float, default=DEFAULT_FOLLOWER_MAX_STEP_RAD
    )
    parser.add_argument(
        "--interpolation-alpha", type=float,
        default=DEFAULT_FOLLOWER_INTERPOLATION_ALPHA,
    )
    parser.add_argument(
        "--joint-scale", action="append", default=[], metavar="ID=SCALE",
        help="Override one post-retarget motor scale; repeat as needed.",
    )
    parser.add_argument(
        "--opening-sign", action="append", default=[], metavar="ID=+1|-1",
        help="Sign of increasing OPENING motion for one motor; repeat as needed.",
    )
    parser.add_argument(
        "--no-configure-follower", action="store_true",
        help="Skip follower register configuration (motion profile is still set).",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrackingConfig:
    scales = dict(DEFAULT_JOINT_COMMAND_SCALES)
    scales.update(
        _parse_key_values(args.joint_scale, value_type=float, name="--joint-scale")
    )
    signs = dict(DEFAULT_OPENING_SIGNS)
    signs.update(
        _parse_key_values(args.opening_sign, value_type=int, name="--opening-sign")
    )
    config = TrackingConfig(
        leader_config=args.leader_config.expanduser(),
        leader_port=args.leader_port,
        follower_config=args.follower_config.expanduser(),
        follower_port=args.follower_port,
        output_root=args.output_root.expanduser(),
        duration_s=args.duration,
        active_motors=tuple(args.active_motors),
        leader_rate_hz=args.leader_rate,
        follower_measure_rate_hz=args.measure_rate,
        follower_command_rate_hz=args.command_rate,
        follower_configure_hardware=not args.no_configure_follower,
        follower_command_scale=args.command_scale,
        follower_max_step_rad=args.max_step,
        follower_interpolation_alpha=args.interpolation_alpha,
        joint_command_scales=scales,
        opening_signs=signs,
        speed_bin_edges_rad_s=tuple(args.speed_bins),
        near_zero_velocity_rad_s=args.near_zero_velocity,
        reversal_exclusion_window_s=args.reversal_window,
        max_command_age_s=args.max_command_age,
        max_leader_age_s=args.max_leader_age,
        make_plots=not args.no_plots,
    )
    validate_config(config)
    return config


def validate_config(config: TrackingConfig) -> None:
    if config.leader_port == config.follower_port:
        raise ValueError("Leader and follower ports must be different")
    if config.duration_s < 0:
        raise ValueError("Duration must be non-negative")
    rates = (config.leader_rate_hz, config.follower_measure_rate_hz,
             config.follower_command_rate_hz)
    if not all(math.isfinite(value) and value > 0 for value in rates):
        raise ValueError("Leader, measurement, and command rates must be positive")
    if not 0 < config.follower_interpolation_alpha <= 1:
        raise ValueError("Interpolation alpha must be in (0, 1]")
    if config.follower_max_step_rad < 0:
        raise ValueError("Maximum step must be non-negative")
    if not math.isfinite(config.follower_command_scale):
        raise ValueError("Command scale must be finite")
    if any(not math.isfinite(float(value))
           for value in config.joint_command_scales.values()):
        raise ValueError("Joint command scales must be finite")
    edges = config.speed_bin_edges_rad_s
    if not edges or any(not math.isfinite(x) or x <= 0 for x in edges):
        raise ValueError("Speed-bin edges must be finite and positive")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("Speed-bin edges must be strictly ascending")
    if config.near_zero_velocity_rad_s < 0 or config.reversal_exclusion_window_s < 0:
        raise ValueError("Velocity and reversal thresholds must be non-negative")
    if config.max_command_age_s <= 0 or config.max_leader_age_s <= 0:
        raise ValueError("Freshness ages must be positive")
    if not config.active_motors or len(set(config.active_motors)) != len(config.active_motors):
        raise ValueError("Active motor IDs must be non-empty and unique")
    missing_signs = sorted(set(config.active_motors) - set(config.opening_signs))
    if missing_signs:
        raise ValueError(f"Missing opening signs for motor IDs {missing_signs}")
    bad_signs = {i: config.opening_signs[i] for i in config.active_motors
                 if config.opening_signs[i] not in (-1, 1)}
    if bad_signs:
        raise ValueError(f"Opening signs must be +1 or -1: {bad_signs}")


def _leader_worker(config: TrackingConfig, shared: SharedCapture, ready: threading.Event,
                   errors: queue.Queue[BaseException]) -> None:
    hand = None
    try:
        from midas_hand_api import MidasHand

        hand_config = load_hand_config(
            config.leader_config, config.leader_port, config.active_motors
        )
        hand = MidasHand(config=hand_config, autoconnect=True)
        verify_connected_hand(hand, role="TRIMORPH leader")
        hand.disable_torque()
        shared.leader_motor_ids = tuple(int(motor_id) for motor_id in hand.motor_ids)
        print(f"TRIMORPH connected (torque disabled): {hand.port}; motors={hand.motor_ids}")
        ready.set()
        period_ns = round(1e9 / config.leader_rate_hz)
        next_ns = time.monotonic_ns()
        sequence = 0
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()
            if now_ns < next_ns:
                shared.shutdown.wait(min((next_ns - now_ns) / 1e9, 0.005))
                continue
            positions = np.asarray(hand.read_pos(), dtype=np.float64)
            sample_ns = time.monotonic_ns()
            if positions.shape != (len(hand.motor_ids),) or not np.isfinite(positions).all():
                raise RuntimeError("TRIMORPH read returned an invalid motor vector")
            shared.set_leader(LeaderSample(sequence, sample_ns, positions))
            sequence += 1
            next_ns += period_ns
            while next_ns < sample_ns:
                next_ns += period_ns
    except BaseException as exc:
        errors.put(exc)
        shared.shutdown.set()
        ready.set()
    finally:
        if hand is not None:
            try:
                hand.disable_torque()
            finally:
                hand.shutdown()


def _append_command_rows(shared: SharedCapture, command: IssuedCommand,
                         motor_ids: Sequence[int]) -> None:
    leader = command.leader_sample
    for index, motor_id in enumerate(motor_ids):
        shared.command_rows.append({
            "command_seq": command.sequence,
            "command_call_start_ns": command.call_start_ns,
            "issued_monotonic_ns": command.issued_ns,
            "command_call_duration_s": (command.issued_ns - command.call_start_ns) / 1e9,
            "leader_seq": "" if leader is None else leader.sequence,
            "leader_monotonic_ns": "" if leader is None else leader.monotonic_ns,
            "joint_id": int(motor_id),
            "joint_name": JOINT_NAMES.get(int(motor_id), f"motor_{motor_id}"),
            "trimorph_position_rad": "" if leader is None else float(leader.positions[index]),
            "q_issued_rad": float(command.target[index]),
            "issued_velocity_rad_s": float(command.velocity_rad_s[index]),
        })


def _follower_worker(config: TrackingConfig, shared: SharedCapture,
                     ready: threading.Event, errors: queue.Queue[BaseException]) -> None:
    hand = None
    try:
        from midas_hand_api import MidasHand

        hand_config = load_follower_config(
            config.follower_config, config.follower_port, config.active_motors
        )
        scales = np.asarray(
            joint_command_scales_for_motor_ids(config.joint_command_scales,
                                               hand_config.motor_ids),
            dtype=np.float64,
        )
        hand = MidasHand(config=hand_config, autoconnect=True)
        verify_connected_hand(hand, role="MIDAS follower")
        follower_motor_ids = tuple(int(motor_id) for motor_id in hand.motor_ids)
        if follower_motor_ids != shared.leader_motor_ids:
            raise RuntimeError(
                "TRIMORPH and MIDAS motor ordering must match exactly; got "
                f"{shared.leader_motor_ids} and {follower_motor_ids}"
            )
        prepare_follower_hand(
            hand, configure_hardware=config.follower_configure_hardware
        )
        last_command = hand.clip_positions(np.asarray(hand.read_pos(), dtype=np.float64))
        call_start_ns = time.monotonic_ns()
        hand.set_positions(last_command, clip=True)
        issued_ns = time.monotonic_ns()
        hand.enable_torque()
        command = IssuedCommand(
            0, call_start_ns, issued_ns, np.asarray(last_command).copy(),
            np.zeros_like(last_command, dtype=np.float64), None,
        )
        history = [command]
        _append_command_rows(shared, command, hand.motor_ids)
        print(
            f"MIDAS connected (position control; force feedback disabled): "
            f"{hand.port}; motors={hand.motor_ids}"
        )
        ready.set()
        while not shared.recording.wait(0.05):
            if shared.shutdown.is_set():
                return
        if shared.recording_start_ns is None:
            raise RuntimeError("Recording start clock was not initialized")
        start_ns = shared.recording_start_ns

        read_period_ns = round(1e9 / config.follower_measure_rate_hz)
        command_period_ns = round(1e9 / config.follower_command_rate_hz)
        next_read_ns = time.monotonic_ns()
        next_command_ns = next_read_ns
        sample_seq = 0
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()
            if now_ns >= next_command_ns:
                leader = shared.latest_leader()
                if leader is not None:
                    desired = hand.clip_positions(
                        leader.positions * config.follower_command_scale * scales
                    )
                    delta = (desired - last_command) * config.follower_interpolation_alpha
                    if config.follower_max_step_rad > 0:
                        delta = np.clip(
                            delta, -config.follower_max_step_rad,
                            config.follower_max_step_rad,
                        )
                    target = np.asarray(
                        hand.clip_positions(last_command + delta), dtype=np.float64
                    )
                    call_start_ns = time.monotonic_ns()
                    hand.set_positions(target, clip=True)
                    issued_ns = time.monotonic_ns()
                    dt_s = (issued_ns - history[-1].issued_ns) / 1e9
                    velocity = ((target - history[-1].target) / dt_s
                                if dt_s > 0 else np.zeros_like(target))
                    command = IssuedCommand(
                        history[-1].sequence + 1, call_start_ns, issued_ns,
                        target.copy(), velocity, leader,
                    )
                    history.append(command)
                    _append_command_rows(shared, command, hand.motor_ids)
                    last_command = target
                next_command_ns += command_period_ns
                while next_command_ns < now_ns:
                    next_command_ns += command_period_ns

            if now_ns >= next_read_ns:
                read_start_ns = time.monotonic_ns()
                positions = np.asarray(hand.read_pos(), dtype=np.float64)
                read_end_ns = time.monotonic_ns()
                if positions.shape != (len(hand.motor_ids),):
                    raise RuntimeError("MIDAS read returned an invalid motor-vector shape")
                measurement_ns = (read_start_ns + read_end_ns) // 2
                aligned = next(
                    (item for item in reversed(history) if item.issued_ns <= measurement_ns),
                    None,
                )
                measurement_valid = bool(np.isfinite(positions).all())
                for index, motor_id in enumerate(hand.motor_ids):
                    leader = None if aligned is None else aligned.leader_sample
                    command_age_s = (float("nan") if aligned is None else
                                     (measurement_ns - aligned.issued_ns) / 1e9)
                    leader_age_s = (float("nan") if leader is None else
                                    (measurement_ns - leader.monotonic_ns) / 1e9)
                    alignment_valid = aligned is not None and command_age_s >= 0
                    sample_fresh = bool(
                        alignment_valid
                        and command_age_s <= config.max_command_age_s
                        and leader is not None
                        and 0 <= leader_age_s <= config.max_leader_age_s
                    )
                    q_issued = float("nan") if aligned is None else float(aligned.target[index])
                    q_robot = float(positions[index])
                    shared.raw_rows.append({
                        "sample_seq": sample_seq,
                        "timestamp_monotonic_ns": measurement_ns,
                        "wall_time_s": time.time(),
                        "elapsed_s": (measurement_ns - start_ns) / 1e9,
                        "read_start_ns": read_start_ns,
                        "read_end_ns": read_end_ns,
                        "read_duration_s": (read_end_ns - read_start_ns) / 1e9,
                        "sample_write_age_s": (time.monotonic_ns() - measurement_ns) / 1e9,
                        "joint_id": int(motor_id),
                        "joint_name": JOINT_NAMES.get(int(motor_id), f"motor_{motor_id}"),
                        "opening_sign": config.opening_signs[int(motor_id)],
                        "leader_seq": "" if leader is None else leader.sequence,
                        "leader_monotonic_ns": "" if leader is None else leader.monotonic_ns,
                        "leader_age_s": leader_age_s,
                        "trimorph_position_rad": (float("nan") if leader is None else
                                                   float(leader.positions[index])),
                        "command_seq": "" if aligned is None else aligned.sequence,
                        "q_issued_monotonic_ns": "" if aligned is None else aligned.issued_ns,
                        "command_age_s": command_age_s,
                        "q_issued_rad": q_issued,
                        "q_robot_rad": q_robot,
                        "raw_error_rad": q_issued - q_robot,
                        "issued_velocity_rad_s": (float("nan") if aligned is None else
                                                  float(aligned.velocity_rad_s[index])),
                        "measurement_valid": measurement_valid and math.isfinite(q_robot),
                        "alignment_valid": alignment_valid,
                        "sample_fresh": sample_fresh,
                    })
                sample_seq += 1
                next_read_ns += read_period_ns
                while next_read_ns < read_end_ns:
                    next_read_ns += read_period_ns

            next_due_ns = min(next_read_ns, next_command_ns)
            shared.shutdown.wait(
                max(0.0, min((next_due_ns - time.monotonic_ns()) / 1e9, 0.005))
            )
    except BaseException as exc:
        errors.put(exc)
        shared.shutdown.set()
        ready.set()
    finally:
        if hand is not None:
            try:
                hand.disable_torque()
            finally:
                hand.shutdown()


def speed_bin_label(speed: float, edges: Sequence[float]) -> str:
    lower = 0.0
    for upper in edges:
        if speed < upper:
            return f"[{lower:g},{upper:g})"
        lower = upper
    return f"[{lower:g},inf)"


def annotate_samples(rows: list[dict[str, object]], config: TrackingConfig) -> None:
    """Add direction/bin/eligibility fields without modifying raw error."""
    by_joint: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_joint.setdefault(int(row["joint_id"]), []).append(row)

    for joint_rows in by_joint.values():
        reversal_times: list[float] = []
        previous_direction = ""
        for row in joint_rows:
            velocity = float(row["issued_velocity_rad_s"])
            signed_opening_velocity = velocity * int(row["opening_sign"])
            if not math.isfinite(velocity) or abs(velocity) < config.near_zero_velocity_rad_s:
                direction = ""
            else:
                direction = "OPENING" if signed_opening_velocity > 0 else "CLOSING"
                if previous_direction and direction != previous_direction:
                    reversal_times.append(float(row["elapsed_s"]))
                previous_direction = direction
            row["direction"] = direction
            row["speed_rad_s"] = abs(velocity)
            row["speed_bin"] = (
                speed_bin_label(abs(velocity), config.speed_bin_edges_rad_s)
                if math.isfinite(velocity) else ""
            )

        for row in joint_rows:
            reasons: list[str] = []
            if not bool(row["measurement_valid"]):
                reasons.append("invalid_measurement")
            if not bool(row["alignment_valid"]):
                reasons.append("invalid_alignment")
            if not bool(row["sample_fresh"]):
                reasons.append("stale_sample")
            velocity = float(row["issued_velocity_rad_s"])
            if not math.isfinite(velocity):
                reasons.append("invalid_velocity")
            elif abs(velocity) < config.near_zero_velocity_rad_s:
                reasons.append("near_zero_velocity")
            elapsed_s = float(row["elapsed_s"])
            if any(abs(elapsed_s - reversal) <= config.reversal_exclusion_window_s
                   for reversal in reversal_times):
                reasons.append("reversal_window")
            row["baseline_eligible"] = not reasons
            row["exclusion_reason"] = ";".join(dict.fromkeys(reasons))


def summarize_samples(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str, str, str], list[float]] = {}
    for row in rows:
        if not bool(row["baseline_eligible"]):
            continue
        key = (int(row["joint_id"]), str(row["joint_name"]),
               str(row["direction"]), str(row["speed_bin"]))
        groups.setdefault(key, []).append(float(row["raw_error_rad"]))

    summary: list[dict[str, object]] = []
    for key in sorted(groups):
        joint_id, joint_name, direction, bin_label = key
        values = np.asarray(groups[key], dtype=np.float64)
        median = float(np.median(values))
        summary.append({
            "joint_id": joint_id,
            "joint_name": joint_name,
            "direction": direction,
            "speed_bin": bin_label,
            "count": int(values.size),
            "median_e_free_rad": median,
            "mad_rad": float(np.median(np.abs(values - median))),
            "p05_rad": float(np.percentile(values, 5)),
            "p25_rad": float(np.percentile(values, 25)),
            "p75_rad": float(np.percentile(values, 75)),
            "p95_rad": float(np.percentile(values, 95)),
        })
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, object]],
               fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", newline="", encoding="utf-8") as file:
        if not names:
            return
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(path: Path, config: TrackingConfig, shared: SharedCapture,
                    summary_count: int, failure: BaseException | None) -> None:
    metadata = {
        "status": "error" if failure is not None else "complete",
        "error": None if failure is None else f"{type(failure).__name__}: {failure}",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "raw_error_definition": "q_issued_aligned_rad - q_robot_rad",
        "error_is_signed_and_not_direction_normalized": True,
        "force_feedback_enabled": False,
        "measurement_timestamp": (
            "midpoint of host read_pos call; the API exposes no device timestamp"
        ),
        "alignment": (
            "newest completed set_positions command whose issued_monotonic_ns is "
            "not later than timestamp_monotonic_ns"
        ),
        "counts": {
            "raw_joint_samples": len(shared.raw_rows),
            "issued_joint_commands": len(shared.command_rows),
            "baseline_eligible": sum(
                bool(row.get("baseline_eligible")) for row in shared.raw_rows
            ),
            "summary_groups": summary_count,
        },
        "config": {
            "leader_config": str(config.leader_config.resolve()),
            "leader_port": config.leader_port,
            "follower_config": str(config.follower_config.resolve()),
            "follower_port": config.follower_port,
            "active_motors": list(config.active_motors),
            "duration_s": config.duration_s,
            "leader_rate_hz": config.leader_rate_hz,
            "follower_measure_rate_hz": config.follower_measure_rate_hz,
            "follower_command_rate_hz": config.follower_command_rate_hz,
            "follower_configure_hardware": config.follower_configure_hardware,
            "follower_command_scale": config.follower_command_scale,
            "follower_max_step_rad": config.follower_max_step_rad,
            "follower_interpolation_alpha": config.follower_interpolation_alpha,
            "joint_command_scales": config.joint_command_scales,
            "opening_signs": config.opening_signs,
            "speed_bin_edges_rad_s": list(config.speed_bin_edges_rad_s),
            "near_zero_velocity_rad_s": config.near_zero_velocity_rad_s,
            "reversal_exclusion_window_s": config.reversal_exclusion_window_s,
            "max_command_age_s": config.max_command_age_s,
            "max_leader_age_s": config.max_leader_age_s,
        },
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def make_plots(rows: Sequence[dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib; install it or pass --no-plots"
        ) from exc

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    joint_ids = sorted({int(row["joint_id"]) for row in rows})
    colors = {"OPENING": "tab:blue", "CLOSING": "tab:orange"}
    for joint_id in joint_ids:
        joint_rows = [row for row in rows if int(row["joint_id"]) == joint_id]
        if not joint_rows:
            continue
        name = str(joint_rows[0]["joint_name"])
        elapsed = np.asarray([float(row["elapsed_s"]) for row in joint_rows])
        issued = np.asarray([float(row["q_issued_rad"]) for row in joint_rows])
        robot = np.asarray([float(row["q_robot_rad"]) for row in joint_rows])
        error = np.asarray([float(row["raw_error_rad"]) for row in joint_rows])
        velocity = np.asarray([float(row["issued_velocity_rad_s"]) for row in joint_rows])

        figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        axes[0, 0].plot(elapsed, issued, label="q_issued", linewidth=1)
        axes[0, 0].plot(elapsed, robot, label="q_robot", linewidth=1)
        axes[0, 0].set(xlabel="time (s)", ylabel="position (rad)",
                       title="Issued target and measured position")
        axes[0, 0].legend()
        axes[0, 1].plot(elapsed, error, linewidth=0.8)
        excluded = np.asarray([not bool(row["baseline_eligible"]) for row in joint_rows])
        axes[0, 1].scatter(elapsed[excluded], error[excluded], s=5, color="0.7",
                           label="excluded")
        axes[0, 1].axhline(0, color="black", linewidth=0.5)
        axes[0, 1].set(xlabel="time (s)", ylabel="q_issued - q_robot (rad)",
                       title="Raw signed tracking error")
        axes[0, 1].legend()

        box_values: list[list[float]] = []
        box_labels: list[str] = []
        for direction in ("OPENING", "CLOSING"):
            mask = np.asarray([
                bool(row["baseline_eligible"]) and row["direction"] == direction
                for row in joint_rows
            ])
            axes[1, 0].scatter(np.abs(velocity[mask]), error[mask], s=8, alpha=0.5,
                               color=colors[direction], label=direction)
            bins = sorted({str(row["speed_bin"]) for row in joint_rows
                           if bool(row["baseline_eligible"]) and row["direction"] == direction})
            for bin_label in bins:
                values = [float(row["raw_error_rad"]) for row in joint_rows
                          if bool(row["baseline_eligible"])
                          and row["direction"] == direction
                          and row["speed_bin"] == bin_label]
                if values:
                    box_values.append(values)
                    box_labels.append(f"{direction[0]} {bin_label}")
        axes[1, 0].axhline(0, color="black", linewidth=0.5)
        axes[1, 0].set(xlabel="|issued velocity| (rad/s)",
                       ylabel="raw error (rad)", title="Error versus speed")
        axes[1, 0].legend()
        if box_values:
            axes[1, 1].boxplot(box_values, tick_labels=box_labels, showfliers=False)
            axes[1, 1].tick_params(axis="x", labelrotation=35)
        axes[1, 1].axhline(0, color="black", linewidth=0.5)
        axes[1, 1].set(ylabel="raw error (rad)",
                       title="Eligible error by direction/speed bin")
        figure.suptitle(f"{name} (motor {joint_id})")
        figure.savefig(plot_dir / f"joint_{joint_id:02d}_{name}.png", dpi=160)
        plt.close(figure)


def _session_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = root / f"session_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def collect(config: TrackingConfig, shared: SharedCapture) -> None:
    errors: queue.Queue[BaseException] = queue.Queue()
    leader_ready = threading.Event()
    follower_ready = threading.Event()
    leader_thread = threading.Thread(
        target=_leader_worker, args=(config, shared, leader_ready, errors), daemon=True
    )
    follower_thread = threading.Thread(
        target=_follower_worker, args=(config, shared, follower_ready, errors), daemon=True
    )
    threads = [leader_thread, follower_thread]
    leader_thread.start()
    try:
        if not leader_ready.wait(60):
            raise TimeoutError("leader did not initialize within 60 seconds")
        if not errors.empty():
            raise errors.get_nowait()
        follower_thread.start()
        if not follower_ready.wait(60):
            raise TimeoutError("follower did not initialize within 60 seconds")
        if not errors.empty():
            raise errors.get_nowait()
        shared.recording_start_ns = time.monotonic_ns()
        shared.recording.set()
        print("Recording free motion. Manually move TRIMORPH; keep MIDAS contact-free.")
        if config.duration_s == 0:
            print("Press Ctrl-C to stop.")
            while not shared.shutdown.wait(0.2):
                if not errors.empty():
                    raise errors.get_nowait()
        else:
            deadline = time.monotonic() + config.duration_s
            while not shared.shutdown.wait(min(0.2, max(0.0, deadline - time.monotonic()))):
                if time.monotonic() >= deadline:
                    break
                if not errors.empty():
                    raise errors.get_nowait()
            if not errors.empty():
                raise errors.get_nowait()
        if not errors.empty():
            raise errors.get_nowait()
    finally:
        shared.shutdown.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=5)


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    output_dir = _session_dir(config.output_root)
    shared = SharedCapture()
    failure: BaseException | None = None
    try:
        collect(config, shared)
    except KeyboardInterrupt:
        print("\nInterrupted; saving the partial recording.")
    except BaseException as exc:
        failure = exc

    annotate_samples(shared.raw_rows, config)
    summary = summarize_samples(shared.raw_rows)
    excluded = [row for row in shared.raw_rows if not bool(row["baseline_eligible"])]
    _write_csv(output_dir / "raw_samples.csv", shared.raw_rows)
    _write_csv(output_dir / "issued_commands.csv", shared.command_rows)
    _write_csv(output_dir / "excluded_samples.csv", excluded,
               fieldnames=list(shared.raw_rows[0].keys()) if shared.raw_rows else None)
    _write_csv(
        output_dir / "e_free_summary.csv", summary,
        fieldnames=("joint_id", "joint_name", "direction", "speed_bin", "count",
                    "median_e_free_rad", "mad_rad", "p05_rad", "p25_rad",
                    "p75_rad", "p95_rad"),
    )
    _write_metadata(output_dir / "session_metadata.json", config, shared,
                    len(summary), failure)
    if config.make_plots and shared.raw_rows:
        make_plots(shared.raw_rows, output_dir)
    eligible_count = sum(bool(row["baseline_eligible"]) for row in shared.raw_rows)
    print(
        f"Saved {len(shared.raw_rows)} joint samples ({eligible_count} baseline-eligible) "
        f"to {output_dir}"
    )
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
