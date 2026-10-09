#!/usr/bin/env python3
"""Replay synchronized PiperX EEF and TRIMORPH leader-hand streams."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from collect_hand_only import (
    DEFAULT_HAND_CONFIG,
    DEFAULT_HAND_MEASURE_RATE_HZ,
    DEFAULT_HAND_PORT,
    DEFAULT_THUMB_MODEL,
    HandOnlyCollectionConfig,
)
from collection.episode import raw_root
from collection.hand_common import (
    load_follower_config,
    load_hand_config,
    verify_connected_hand,
)
from collection.thumb_retargeting import (
    THUMB_RETARGET_MOTOR_IDS,
    ThumbRetargetModel,
    build_follower_target,
    load_thumb_retarget_model,
)


DATASET_ROOT = Path("data_trimorph")
RAW_EEF_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
REPLAY_HZ = 30.0
CURRENT_TELEMETRY_HZ = 10.0
PREPOSITION_SECONDS = 3.0
SETTLE_SECONDS = 1.0
PIPER_PREPOSITION_SPEED_PERCENT = 20
PIPER_REPLAY_SPEED_PERCENT = 100
PIPER_INTERFACE = "socketcan"
PIPER_CHANNEL = "can0"
PIPER_FIRMWARE = "v183"
PIPER_INSTALLATION_POS = "right"
PIPER_PAYLOAD = "half"
PURE_POSITION_OPERATING_MODE = 3
ACTION_SCALE = 1.1


def advance_current_sample_schedule(
    elapsed_s: float,
    next_sample_s: float,
    sample_period_s: float,
) -> tuple[bool, float]:
    """Return whether to sample now and the following fixed-rate deadline."""

    if elapsed_s + 1e-12 < next_sample_s:
        return False, next_sample_s
    while next_sample_s <= elapsed_s + 1e-12:
        next_sample_s += sample_period_s
    return True, next_sample_s


class ReplayCurrentRecorder:
    """Collect replay telemetry in gen_graphs_from_csvs.py's native schema."""

    def __init__(self, motor_ids: tuple[int, ...]) -> None:
        self.motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
        self.rows: list[dict[str, float | int]] = []
        self._previous_time_s: float | None = None
        self._previous_targets: np.ndarray | None = None

    def sample(
        self,
        hand,
        target_positions_rad: np.ndarray,
        *,
        elapsed_s: float,
    ) -> None:
        targets = np.asarray(target_positions_rad, dtype=np.float64)
        actual_positions, actual_velocities, currents = hand.read_pos_vel_cur()
        actual_positions = np.asarray(actual_positions, dtype=np.float64)
        actual_velocities = np.asarray(actual_velocities, dtype=np.float64)
        currents = np.asarray(currents, dtype=np.float64)
        expected_shape = (len(self.motor_ids),)
        if any(
            values.shape != expected_shape
            for values in (targets, actual_positions, actual_velocities, currents)
        ):
            raise RuntimeError("Replay telemetry vectors do not match the motor count")
        if not all(
            np.isfinite(values).all()
            for values in (targets, actual_positions, actual_velocities, currents)
        ):
            raise RuntimeError("Replay telemetry contains NaN or Inf")

        if self._previous_time_s is None or self._previous_targets is None:
            target_velocities = np.zeros_like(targets)
        else:
            delta_s = elapsed_s - self._previous_time_s
            if delta_s <= 0.0:
                raise RuntimeError("Replay telemetry timestamps must increase")
            target_velocities = (targets - self._previous_targets) / delta_s

        row: dict[str, float | int] = {"trial": 1, "elapsed_s": elapsed_s}
        for index, motor_id in enumerate(self.motor_ids):
            row.update(
                {
                    f"motor_{motor_id}_current_ma": float(currents[index]),
                    f"motor_{motor_id}_target_position_rad": float(targets[index]),
                    f"motor_{motor_id}_actual_position_rad": float(
                        actual_positions[index]
                    ),
                    f"motor_{motor_id}_target_velocity_rad_s": float(
                        target_velocities[index]
                    ),
                    f"motor_{motor_id}_actual_velocity_rad_s": float(
                        actual_velocities[index]
                    ),
                }
            )
        self.rows.append(row)
        self._previous_time_s = elapsed_s
        self._previous_targets = targets.copy()


def output_current_csv(recorder: ReplayCurrentRecorder, output: Path) -> None:
    if not recorder.rows:
        raise RuntimeError("No replay current samples were recorded")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["trial", "elapsed_s"]
    for motor_id in recorder.motor_ids:
        fieldnames.extend(
            (
                f"motor_{motor_id}_current_ma",
                f"motor_{motor_id}_target_position_rad",
                f"motor_{motor_id}_actual_position_rad",
                f"motor_{motor_id}_target_velocity_rad_s",
                f"motor_{motor_id}_actual_velocity_rad_s",
            )
        )
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(recorder.rows)
    print(f"Saved replay current telemetry CSV to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one raw PiperX/TRIMORPH episode (latest by default)."
    )
    parser.add_argument(
        "episode",
        nargs="?",
        help="Episode directory, numeric index, or episode_XXXXXX name.",
    )
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--hand-config", "--leader-config", dest="hand_config")
    parser.add_argument("--hand-port", "--leader-port", dest="hand_port")
    parser.add_argument(
        "--use-follower",
        action="store_true",
        help=(
            "Treat --hand-config and --hand-port as a MIDAS follower, applying "
            "normal action scaling plus the joints 2-3 thumb model."
        ),
    )
    parser.add_argument(
        "--thumb-model",
        type=Path,
        default=DEFAULT_THUMB_MODEL,
        help=(
            "Joints 2-3 TRIMORPH-to-MIDAS model used only with --use-follower "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument("--rate-hz", type=float, default=REPLAY_HZ)
    parser.add_argument("--no-preposition", action="store_true")
    parser.add_argument("--preposition-seconds", type=float, default=PREPOSITION_SECONDS)
    parser.add_argument("--settle-seconds", type=float, default=SETTLE_SECONDS)
    parser.add_argument("--hand-only", action="store_true")
    parser.add_argument("--arm-only", action="store_true")
    parser.add_argument(
        "--output-current",
        type=Path,
        metavar="PATH",
        help=(
            "Measure hand motor current during replay and write graph-compatible "
            "telemetry to PATH. No telemetry is saved when omitted."
        ),
    )
    parser.add_argument(
        "--no-piper-enable",
        action="store_true",
        help="Do not enable Piper; also permits replay of episodes with an empty piper.csv.",
    )
    return parser.parse_args()


def resolve_episode(dataset_root: Path, episode: str | None) -> Path:
    root = raw_root(dataset_root)
    if episode is None:
        episodes = sorted(path for path in root.glob("episode_*") if path.is_dir())
        if not episodes:
            raise FileNotFoundError(f"No raw episodes found under {root}")
        return episodes[-1]

    candidate = Path(episode).expanduser()
    if candidate.is_dir():
        return candidate
    if episode.isdigit():
        candidate = root / f"episode_{int(episode):06d}"
    elif episode.startswith("episode_"):
        candidate = root / episode
    else:
        candidate = root / f"episode_{episode}"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Episode not found: {candidate}")
    return candidate


def resolve_leader_hand_path(episode_dir: Path) -> Path:
    path = episode_dir / "leader_hand.csv"
    legacy_path = episode_dir / "proprioception.csv"
    if not path.exists() and legacy_path.exists():
        return legacy_path
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def hand_motor_ids(path: Path) -> tuple[int, ...]:
    with path.open("r", newline="") as csv_file:
        fieldnames = csv.DictReader(csv_file).fieldnames or []
    motor_ids = tuple(
        int(field.removeprefix("pos_"))
        for field in fieldnames
        if field.startswith("pos_")
    )
    if not motor_ids:
        raise ValueError(f"No motor position columns in {path}")
    return motor_ids


def load_stream(
    path: Path,
    fields: tuple[str, ...],
    *,
    allow_empty: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    rows = read_csv_rows(path)
    if not rows:
        if allow_empty:
            return None
        raise ValueError(f"No samples in {path}")
    times = np.asarray([float(row["elapsed_s"]) for row in rows], dtype=np.float64)
    values = np.asarray(
        [[float(row[field]) for field in fields] for row in rows], dtype=np.float64
    )
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError(f"NaN or Inf in {path}")
    order = np.argsort(times, kind="stable")
    times = times[order]
    values = values[order]
    if np.any(np.diff(times) < 0):
        raise ValueError(f"Non-monotonic timestamps in {path}")
    return times, values


def load_hand_stream(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stream = load_stream(
        path, tuple(f"pos_{motor_id}" for motor_id in hand_motor_ids(path))
    )
    assert stream is not None
    return stream


def scale_hand_stream(
    stream: tuple[np.ndarray, np.ndarray], scale: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    times, positions = stream
    return times, positions * scale


def prepare_hand_stream_for_replay(
    stream: tuple[np.ndarray, np.ndarray],
    motor_ids: tuple[int, ...],
    *,
    use_follower: bool,
    command_scale: float,
    joint_command_scales: np.ndarray,
    thumb_model: ThumbRetargetModel | None,
    joint_command_offsets: np.ndarray | None = None,
    action_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build replay targets and apply the global action scale to every joint."""

    if not use_follower:
        targets = stream
    else:
        times, positions = stream
        retargeted_positions = np.asarray(
            [
                build_follower_target(
                    position,
                    motor_ids,
                    command_scale=command_scale,
                    joint_command_scales=joint_command_scales,
                    joint_command_offsets=joint_command_offsets,
                    thumb_model=thumb_model,
                )
                for position in positions
            ],
            dtype=np.float64,
        )
        targets = times, retargeted_positions
    return scale_hand_stream(targets, action_scale)


def config_values_for_motor_ids(
    values_by_motor_id: dict[int, float],
    motor_ids: tuple[int, ...],
    *,
    default: float,
) -> np.ndarray:
    """Resolve motor-ID keyed hand-only tuning in recorded column order."""

    return np.asarray(
        [values_by_motor_id.get(int(motor_id), default) for motor_id in motor_ids],
        dtype=np.float64,
    )


def build_replay_config(
    args: argparse.Namespace,
    active_motors: tuple[int, ...] | None,
) -> HandOnlyCollectionConfig:
    """Build replay settings from the hand-only collection configuration."""

    return HandOnlyCollectionConfig(
        dataset_root=Path(args.dataset_root).expanduser(),
        hand_config=(
            Path(args.hand_config).expanduser()
            if args.hand_config is not None
            else DEFAULT_HAND_CONFIG
        ),
        hand_port=args.hand_port or DEFAULT_HAND_PORT,
        hand_measure_rate_hz=DEFAULT_HAND_MEASURE_RATE_HZ,
        warmup_time=0.0,
        episode_time=0.0,
        active_motors=list(active_motors or ()),
    )


def load_piper_stream(
    path: Path, *, allow_empty: bool = False
) -> tuple[np.ndarray, np.ndarray] | None:
    return load_stream(path, RAW_EEF_FIELDS, allow_empty=allow_empty)


def sample_stream(times: np.ndarray, values: np.ndarray, query_t: float) -> np.ndarray:
    return np.asarray(
        [np.interp(query_t, times, values[:, column]) for column in range(values.shape[1])],
        dtype=np.float64,
    )


def replay_interval(*streams: tuple[np.ndarray, np.ndarray] | None) -> tuple[float, float]:
    active = [stream for stream in streams if stream is not None]
    if not active:
        raise ValueError("Nothing selected for replay")
    start_s = max(float(stream[0][0]) for stream in active)
    end_s = min(float(stream[0][-1]) for stream in active)
    if end_s < start_s:
        raise ValueError("Selected streams do not overlap in episode time")
    return start_s, end_s


def build_replay_hand_config(
    hand_config: Path,
    hand_port: str,
    active_motors: tuple[int, ...],
    *,
    use_follower: bool,
):
    """Load a narrowed hand configuration using pure position control.

    Collection and teleoperation retain the configured operating mode, while
    replay deliberately uses Dynamixel mode 3 so the recorded positions are
    not softened by the goal-current limit used in mode 5.
    """

    loader = load_follower_config if use_follower else load_hand_config
    config = loader(hand_config, hand_port, active_motors)
    return replace(config, operating_mode=PURE_POSITION_OPERATING_MODE)


def move_to_first_pose(hand, api, hand_stream, piper_stream, start_s: float, args) -> None:
    if args.no_preposition:
        return
    print("Moving to first synchronized pose...")
    if api is not None:
        api.set_speed_percent(PIPER_PREPOSITION_SPEED_PERCENT)
    first_hand = sample_stream(*hand_stream, start_s) if hand_stream is not None else None
    first_eef = sample_stream(*piper_stream, start_s) if piper_stream is not None else None
    end_t = time.monotonic() + args.preposition_seconds
    while time.monotonic() < end_t:
        if hand is not None and first_hand is not None:
            hand.set_positions(first_hand, clip=True)
        if api is not None and first_eef is not None:
            api.move_eef(first_eef)
        time.sleep(0.05)
    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)


def main() -> int:
    args = parse_args()
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be greater than 0")
    if args.preposition_seconds < 0 or args.settle_seconds < 0:
        raise ValueError("Preposition and settle durations must be non-negative")
    if args.hand_only and args.arm_only:
        raise ValueError("--hand-only and --arm-only are mutually exclusive")
    if args.output_current is not None and args.arm_only:
        raise ValueError("--output-current cannot be used with --arm-only")

    replay_hand = not args.arm_only
    replay_arm = not args.hand_only
    episode_dir = resolve_episode(Path(args.dataset_root).expanduser(), args.episode)
    hand_path = resolve_leader_hand_path(episode_dir)
    active_motors = hand_motor_ids(hand_path) if replay_hand else None
    hand_stream = load_hand_stream(hand_path) if replay_hand else None
    piper_stream = (
        load_piper_stream(
            episode_dir / "piper.csv", allow_empty=args.no_piper_enable
        )
        if replay_arm
        else None
    )
    if replay_arm and piper_stream is None:
        replay_arm = False
        print("piper.csv is empty; replaying hand only.")
    start_s, end_s = replay_interval(hand_stream, piper_stream)
    config = build_replay_config(args, active_motors)
    if hand_stream is not None:
        replay_motor_ids = active_motors or ()
        joint_scales = config_values_for_motor_ids(
            config.follower_joint_command_scales,
            replay_motor_ids,
            default=1.0,
        )
        joint_offsets = config_values_for_motor_ids(
            config.follower_joint_command_offsets,
            replay_motor_ids,
            default=0.0,
        )
        active_motor_set = set(active_motors or ())
        needs_thumb_model = bool(
            active_motor_set & set(THUMB_RETARGET_MOTOR_IDS)
        )
        thumb_model = (
            load_thumb_retarget_model(args.thumb_model.expanduser())
            if args.use_follower and needs_thumb_model
            else None
        )
        hand_stream = prepare_hand_stream_for_replay(
            hand_stream,
            active_motors or (),
            use_follower=args.use_follower,
            command_scale=config.follower_command_scale,
            joint_command_scales=joint_scales,
            joint_command_offsets=joint_offsets,
            thumb_model=thumb_model,
            action_scale=ACTION_SCALE,
        )

    hand = None
    api = None
    current_recorder: ReplayCurrentRecorder | None = None
    try:
        if replay_hand:
            from midas_hand_api import MidasHand

            replay_hand_config = build_replay_hand_config(
                config.hand_config,
                config.hand_port,
                active_motors or (),
                use_follower=args.use_follower,
            )
            hand = MidasHand(config=replay_hand_config, autoconnect=True)
            verify_connected_hand(
                hand, role="follower" if args.use_follower else "replay"
            )
            hand.configure(enable_torque=False)
            print("Hand operating mode: pure position control (mode 3)")
            hand.set_motion_profile(
                profile_velocity_rad_s=config.follower_profile_velocity_rad_s,
                profile_acceleration_rad_s2=(
                    config.follower_profile_acceleration_rad_s2
                ),
            )
            hand.set_positions(hand.read_pos(), clip=True)
            hand.enable_torque()
            if args.output_current is not None:
                current_recorder = ReplayCurrentRecorder(tuple(hand.motor_ids))
            if args.use_follower:
                print(f"MIDAS hand connected on {hand.port}")
            else:
                print(f"TRIMORPH hand connected on {hand.port}")

        if replay_arm:
            from scripts.piper_api.piper_api import PiperAPI

            api = PiperAPI(
                firmware=PIPER_FIRMWARE,
                interface=PIPER_INTERFACE,
                channel=PIPER_CHANNEL,
                installation_pos=PIPER_INSTALLATION_POS,
                payload=PIPER_PAYLOAD,
            )
            api.connect()
            print("Piper connected:", api.is_ok(), "receive FPS:", api.get_fps())
            if not args.no_piper_enable and not api.enable():
                raise RuntimeError("Timed out while enabling Piper")

        move_to_first_pose(hand, api, hand_stream, piper_stream, start_s, args)
        if api is not None:
            api.set_speed_percent(PIPER_REPLAY_SPEED_PERCENT)

        period_s = 1.0 / args.rate_hz
        current_sample_period_s = 1.0 / CURRENT_TELEMETRY_HZ
        next_current_sample_s = 0.0
        wall_start = time.monotonic()
        tick = 0
        print(
            f"Replaying {episode_dir} from {start_s:.3f}s to {end_s:.3f}s "
            f"at {args.rate_hz:.1f} Hz."
        )
        while True:
            episode_t = min(start_s + tick * period_s, end_s)
            if hand is not None and hand_stream is not None:
                hand_target = hand.clip_positions(
                    sample_stream(*hand_stream, episode_t)
                )
                hand.set_positions(hand_target, clip=True)
                replay_elapsed_s = episode_t - start_s
                sample_current, next_current_sample_s = (
                    advance_current_sample_schedule(
                        replay_elapsed_s,
                        next_current_sample_s,
                        current_sample_period_s,
                    )
                )
                if current_recorder is not None and sample_current:
                    current_recorder.sample(
                        hand,
                        hand_target,
                        elapsed_s=replay_elapsed_s,
                    )
            if api is not None and piper_stream is not None:
                api.move_eef(sample_stream(*piper_stream, episode_t))
            if episode_t >= end_s:
                break
            tick += 1
            sleep_s = wall_start + tick * period_s - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
        if current_recorder is not None:
            output_current_csv(current_recorder, args.output_current)
        print("Replay complete.")
        return 0
    except KeyboardInterrupt:
        if current_recorder is not None and current_recorder.rows:
            output_current_csv(current_recorder, args.output_current)
        print("\nReplay interrupted.")
        return 130
    finally:
        if api is not None:
            api.disconnect()
        if hand is not None:
            hand.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
