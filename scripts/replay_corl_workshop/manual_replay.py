#!/usr/bin/env python3
"""Replay alternating PiperX and TRIMORPH CSV trajectories."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_COLLECTION_ROOT = REPO_ROOT / "data_collection"
for import_root in (REPO_ROOT, DATA_COLLECTION_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from collection.hand_common import load_hand_config, verify_connected_hand  # noqa: E402
from scripts.piper_api.foot_pedal import (  # noqa: E402
    FootPedalTrigger,
    format_input_devices,
    list_input_devices,
)
from scripts.piper_api.piper_api import PiperAPI  # noqa: E402


PIPER_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
POSITION_COLUMN = re.compile(r"pos_(\d+)\Z")
DEFAULT_HAND_CONFIG = Path("config/hand.yaml")
DEFAULT_HAND_PORT = "/dev/serial/by-id/REPLACE_WITH_TRIMORPH_HAND"
DEFAULT_RATE_HZ = 30.0
DEFAULT_PREPOSITION_SECONDS = 3.0
DEFAULT_SETTLE_SECONDS = 1.0
DEFAULT_HAND_PROFILE_VELOCITY_RAD_S = 3.0
DEFAULT_HAND_PROFILE_ACCELERATION_RAD_S2 = 50.0
PEDAL_POLL_SECONDS = 0.01
INITIAL_HAND_JOINT_ANGLES_RAD = {
    0: -0.01,
    1: -0.733864183,
    2: 1.316699251,
    3: 0.715970865,
    4: -0.023689328,
    5: -0.276737908,
    6: -0.082523867,
    7: -0.016019424,
    8: 0.039262134,
    9: -0.095999810,
    10: -0.016019424,
    11: -0.149417502,
    12: 0.053922750,
}


@dataclass(frozen=True)
class PiperTrajectory:
    path: Path
    times: np.ndarray
    poses: np.ndarray


@dataclass(frozen=True)
class TrimorphTrajectory:
    path: Path
    times: np.ndarray
    positions: np.ndarray
    motor_ids: tuple[int, ...]


Trajectory = PiperTrajectory | TrimorphTrajectory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay newline-listed CSVs in Piper/Trimorph alternating order, "
            "waiting for a foot-pedal press after every CSV."
        )
    )
    parser.add_argument(
        "trajectory_list",
        nargs="?",
        type=Path,
        help="Text file containing one CSV path per line, starting with Piper.",
    )
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--no-preposition", action="store_true")
    parser.add_argument(
        "--preposition-seconds", type=float, default=DEFAULT_PREPOSITION_SECONDS
    )
    parser.add_argument("--settle-seconds", type=float, default=DEFAULT_SETTLE_SECONDS)
    parser.add_argument("--preposition-speed-percent", type=int, default=20)
    parser.add_argument("--replay-speed-percent", type=int, default=100)

    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can0")
    parser.add_argument(
        "--firmware", choices=("default", "v183", "v188"), default="v183"
    )
    parser.add_argument(
        "--installation-pos",
        choices=("unchanged", "horizontal", "left", "right"),
        default="right",
    )
    parser.add_argument(
        "--payload",
        choices=("unchanged", "empty", "half", "full"),
        default="half",
    )

    parser.add_argument("--hand-config", type=Path, default=DEFAULT_HAND_CONFIG)
    parser.add_argument("--hand-port", default=DEFAULT_HAND_PORT)
    parser.add_argument(
        "--hand-profile-velocity-rad-s",
        type=float,
        default=DEFAULT_HAND_PROFILE_VELOCITY_RAD_S,
    )
    parser.add_argument(
        "--hand-profile-acceleration-rad-s2",
        type=float,
        default=DEFAULT_HAND_PROFILE_ACCELERATION_RAD_S2,
    )

    parser.add_argument("--foot-pedal-device")
    parser.add_argument("--foot-pedal-key-code", type=int)
    parser.add_argument(
        "--foot-pedal-debounce-seconds", type=float, default=0.5
    )
    parser.add_argument("--list-input-devices", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.trajectory_list is None:
        raise ValueError("trajectory_list is required")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be greater than 0")
    if args.preposition_seconds < 0 or args.settle_seconds < 0:
        raise ValueError("Preposition and settle durations must be non-negative")
    for name in ("preposition_speed_percent", "replay_speed_percent"):
        if not 0 <= getattr(args, name) <= 100:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 100")
    if args.hand_profile_velocity_rad_s <= 0:
        raise ValueError("--hand-profile-velocity-rad-s must be greater than 0")
    if args.hand_profile_acceleration_rad_s2 <= 0:
        raise ValueError("--hand-profile-acceleration-rad-s2 must be greater than 0")
    if args.foot_pedal_debounce_seconds < 0:
        raise ValueError("--foot-pedal-debounce-seconds must be non-negative")
    if not args.hand_port or "REPLACE_WITH" in args.hand_port:
        raise ValueError("Set an explicit --hand-port for the TRIMORPH hand")


def load_trajectory_paths(list_path: Path) -> list[Path]:
    list_path = list_path.expanduser().resolve()
    if not list_path.is_file():
        raise FileNotFoundError(f"Trajectory list not found: {list_path}")

    paths: list[Path] = []
    for line_number, raw_line in enumerate(
        list_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = list_path.parent / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Trajectory path on line {line_number} does not exist: {candidate}"
            )
        paths.append(candidate)

    if not paths:
        raise ValueError(f"No trajectory CSV paths in {list_path}")
    return paths


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or ())
            rows = list(reader)
    except (csv.Error, UnicodeError) as exc:
        raise ValueError(f"Could not parse CSV {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"No samples in {path}")
    return fieldnames, rows


def _numeric_matrix(
    rows: list[dict[str, str]], fields: Sequence[str], path: Path
) -> np.ndarray:
    values: list[list[float]] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            values.append([float(row[field]) for field in fields])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid numeric value in {path} at CSV row {row_number}"
            ) from exc
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"NaN or Inf in {path}")
    return result


def _validate_times(times: np.ndarray, path: Path) -> None:
    if np.any(np.diff(times) < 0):
        raise ValueError(f"Non-monotonic timestamps in {path}")


def load_piper_trajectory(path: Path) -> PiperTrajectory:
    fieldnames, rows = _read_csv(path)
    required = ("elapsed_s", *PIPER_FIELDS)
    missing = [field for field in required if field not in fieldnames]
    if missing:
        raise ValueError(f"Missing Piper columns in {path}: {', '.join(missing)}")
    matrix = _numeric_matrix(rows, required, path)
    times, poses = matrix[:, 0], matrix[:, 1:]
    _validate_times(times, path)
    return PiperTrajectory(path=path, times=times, poses=poses)


def load_trimorph_trajectory(path: Path) -> TrimorphTrajectory:
    fieldnames, rows = _read_csv(path)
    if "elapsed_s" not in fieldnames:
        raise ValueError(f"Missing Trimorph column in {path}: elapsed_s")

    position_fields: list[str] = []
    motor_ids: list[int] = []
    for field in fieldnames:
        match = POSITION_COLUMN.fullmatch(field)
        if match is not None:
            position_fields.append(field)
            motor_ids.append(int(match.group(1)))
    if not position_fields:
        raise ValueError(f"No Trimorph pos_<motor_id> columns in {path}")
    if len(set(motor_ids)) != len(motor_ids):
        raise ValueError(f"Duplicate Trimorph motor position columns in {path}")

    matrix = _numeric_matrix(rows, ("elapsed_s", *position_fields), path)
    times, positions = matrix[:, 0], matrix[:, 1:]
    _validate_times(times, path)
    return TrimorphTrajectory(
        path=path,
        times=times,
        positions=positions,
        motor_ids=tuple(motor_ids),
    )


def load_alternating_trajectories(paths: Sequence[Path]) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for index, path in enumerate(paths):
        expected = "Piper" if index % 2 == 0 else "Trimorph"
        try:
            trajectory = (
                load_piper_trajectory(path)
                if index % 2 == 0
                else load_trimorph_trajectory(path)
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Invalid {expected} CSV at entry {index + 1} ({path}): {exc}"
            ) from exc
        trajectories.append(trajectory)
    return trajectories


def validate_trimorph_motor_ids(
    trajectories: Sequence[Trajectory], configured_motor_ids: Sequence[int]
) -> None:
    configured = {int(motor_id) for motor_id in configured_motor_ids}
    for trajectory in trajectories:
        if not isinstance(trajectory, TrimorphTrajectory):
            continue
        unknown = sorted(set(trajectory.motor_ids) - configured)
        if unknown:
            raise ValueError(
                f"Trimorph CSV {trajectory.path} references motor IDs not present "
                f"in the hand configuration: {unknown}"
            )


def sample_trajectory(
    times: np.ndarray, values: np.ndarray, query_s: float
) -> np.ndarray:
    if query_s <= times[0]:
        return values[0].copy()
    if query_s >= times[-1]:
        return values[-1].copy()
    return np.asarray(
        [np.interp(query_s, times, values[:, column]) for column in range(values.shape[1])],
        dtype=np.float64,
    )


def merge_hand_target(
    hold_target: np.ndarray,
    configured_motor_ids: Sequence[int],
    trajectory_motor_ids: Sequence[int],
    trajectory_positions: Sequence[float],
) -> np.ndarray:
    target = np.asarray(hold_target, dtype=np.float64).copy()
    index_by_motor_id = {
        int(motor_id): index for index, motor_id in enumerate(configured_motor_ids)
    }
    for motor_id, position in zip(trajectory_motor_ids, trajectory_positions):
        target[index_by_motor_id[int(motor_id)]] = float(position)
    return target


def _timed_replay(
    times: np.ndarray,
    values: np.ndarray,
    rate_hz: float,
    command,
) -> None:
    start_s = float(times[0])
    end_s = float(times[-1])
    duration_s = end_s - start_s
    period_s = 1.0 / rate_hz
    wall_start = time.monotonic()
    tick = 0
    while True:
        elapsed_s = min(tick * period_s, duration_s)
        command(sample_trajectory(times, values, start_s + elapsed_s))
        if elapsed_s >= duration_s:
            return
        tick += 1
        sleep_s = wall_start + tick * period_s - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)


def _preposition(command, target: np.ndarray, args: argparse.Namespace) -> None:
    if args.no_preposition:
        return
    deadline = time.monotonic() + args.preposition_seconds
    while time.monotonic() < deadline:
        command(target)
        time.sleep(0.05)
    if args.settle_seconds:
        time.sleep(args.settle_seconds)


def replay_piper(
    api: PiperAPI, trajectory: PiperTrajectory, args: argparse.Namespace
) -> None:
    if not args.no_preposition:
        print(f"Prepositioning Piper for {trajectory.path}...")
        api.set_speed_percent(args.preposition_speed_percent)
        _preposition(api.move_eef, trajectory.poses[0], args)
    api.set_speed_percent(args.replay_speed_percent)
    print(f"Replaying Piper CSV: {trajectory.path}")
    _timed_replay(trajectory.times, trajectory.poses, args.rate_hz, api.move_eef)


def replay_trimorph(
    hand,
    trajectory: TrimorphTrajectory,
    hold_target: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    motor_ids = tuple(int(motor_id) for motor_id in hand.motor_ids)

    def command(partial_target: np.ndarray) -> None:
        nonlocal hold_target
        merged = merge_hand_target(
            hold_target, motor_ids, trajectory.motor_ids, partial_target
        )
        hold_target = np.asarray(hand.clip_positions(merged), dtype=np.float64)
        hand.set_positions(hold_target, clip=True)

    if not args.no_preposition:
        print(f"Prepositioning TRIMORPH for {trajectory.path}...")
        _preposition(command, trajectory.positions[0], args)
    print(f"Replaying Trimorph CSV: {trajectory.path}")
    _timed_replay(trajectory.times, trajectory.positions, args.rate_hz, command)
    return hold_target


def wait_for_fresh_pedal_press(
    pedal: FootPedalTrigger, *, completed: int, total: int, path: Path
) -> None:
    pedal.poll()  # Discard presses made before this trajectory completed.
    print(
        f"Completed {completed}/{total}: {path}. "
        "Press the foot pedal to continue."
    )
    while True:
        events = pedal.poll()
        if events:
            print(events[-1].message)
            return
        time.sleep(PEDAL_POLL_SECONDS)


def wait_for_initial_position_press(pedal: FootPedalTrigger) -> None:
    pedal.poll()  # Discard input received before both devices became ready.
    print("Press the foot pedal to move TRIMORPH to its initial joint position.")
    while True:
        events = pedal.poll()
        if events:
            print(events[-1].message)
            return
        time.sleep(PEDAL_POLL_SECONDS)


def wait_for_first_trajectory_press(pedal: FootPedalTrigger) -> None:
    pedal.poll()  # Discard input received while the hand was moving.
    print("Press the foot pedal to start the first Piper trajectory.")
    while True:
        events = pedal.poll()
        if events:
            print(events[-1].message)
            return
        time.sleep(PEDAL_POLL_SECONDS)


def initial_hand_target(motor_ids: Sequence[int]) -> np.ndarray:
    missing = sorted(
        int(motor_id)
        for motor_id in motor_ids
        if int(motor_id) not in INITIAL_HAND_JOINT_ANGLES_RAD
    )
    if missing:
        raise ValueError(
            "No initial hand joint angle is configured for motor IDs: "
            f"{missing}"
        )
    return np.asarray(
        [INITIAL_HAND_JOINT_ANGLES_RAD[int(motor_id)] for motor_id in motor_ids],
        dtype=np.float64,
    )


def move_hand_to_initial_position(
    hand, motor_ids: Sequence[int], args: argparse.Namespace
) -> np.ndarray:
    target = initial_hand_target(motor_ids)
    clipped = np.asarray(hand.clip_positions(target), dtype=np.float64)
    if clipped.shape != target.shape or not np.allclose(clipped, target):
        raise RuntimeError(
            "Configured hand limits would clip the initial joint position; "
            "refusing to move"
        )

    print("Moving TRIMORPH to its initial joint position...")
    hand.set_positions(target, clip=True)
    deadline = time.monotonic() + args.preposition_seconds
    while time.monotonic() < deadline:
        hand.set_positions(target, clip=True)
        time.sleep(0.05)
    if args.settle_seconds:
        time.sleep(args.settle_seconds)
    print("TRIMORPH initial-position command complete.")
    return target


def _create_hand(hand_config):
    from midas_hand_api import MidasHand

    return MidasHand(config=hand_config, autoconnect=True)


def run(args: argparse.Namespace) -> int:
    if args.list_input_devices:
        print(format_input_devices(list_input_devices()))
        return 0

    validate_args(args)
    paths = load_trajectory_paths(args.trajectory_list)
    trajectories = load_alternating_trajectories(paths)

    # Loading and cross-checking the full configuration is still preflight: no
    # serial or CAN device is opened until every CSV has passed validation.
    hand_config = load_hand_config(
        args.hand_config.expanduser(), args.hand_port, active_motors=None
    )
    validate_trimorph_motor_ids(trajectories, hand_config.motor_ids)
    initial_hand_target(hand_config.motor_ids)

    pedal = FootPedalTrigger(
        args.foot_pedal_device,
        key_code=args.foot_pedal_key_code,
        debounce_s=args.foot_pedal_debounce_seconds,
    )
    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
        installation_pos=args.installation_pos,
        payload=args.payload,
    )
    hand = None
    api_connected = False
    try:
        pedal.start()
        print(f"Foot pedal input: {pedal.active_device}")

        hand = _create_hand(hand_config)
        verify_connected_hand(hand, role="TRIMORPH")
        hand.configure(enable_torque=False)

        api_connected = True
        api.connect()
        print(f"Piper connected: {api.is_ok()} (receive FPS: {api.get_fps():.1f})")
        if not api.is_ok():
            raise RuntimeError("Piper is not responding on the CAN bus")
        if not api.enable():
            raise RuntimeError("Timed out while enabling Piper")
        api.set_position_mode()

        hand.set_motion_profile(
            profile_velocity_rad_s=args.hand_profile_velocity_rad_s,
            profile_acceleration_rad_s2=args.hand_profile_acceleration_rad_s2,
        )
        hold_target = np.asarray(hand.read_pos(), dtype=np.float64)
        if hold_target.shape != (len(hand.motor_ids),) or not np.isfinite(
            hold_target
        ).all():
            raise RuntimeError("TRIMORPH returned an invalid initial joint position")
        hold_target = np.asarray(hand.clip_positions(hold_target), dtype=np.float64)
        hand.set_positions(hold_target, clip=True)
        hand.enable_torque()
        print(f"TRIMORPH connected on {hand.port}; torque enabled at current pose")

        wait_for_initial_position_press(pedal)
        hold_target = move_hand_to_initial_position(
            hand, tuple(int(motor_id) for motor_id in hand.motor_ids), args
        )
        wait_for_first_trajectory_press(pedal)

        total = len(trajectories)
        for index, trajectory in enumerate(trajectories, start=1):
            if isinstance(trajectory, PiperTrajectory):
                replay_piper(api, trajectory, args)
            else:
                hold_target = replay_trimorph(
                    hand, trajectory, hold_target, args
                )
            wait_for_fresh_pedal_press(
                pedal, completed=index, total=total, path=trajectory.path
            )
        print("All trajectories replayed and acknowledged.")
        return 0
    except KeyboardInterrupt:
        print("\nReplay interrupted.")
        return 130
    finally:
        try:
            pedal.stop()
        finally:
            try:
                if hand is not None:
                    hand.shutdown()
            finally:
                if api_connected:
                    api.disconnect()


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
