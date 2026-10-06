#!/usr/bin/env python3
"""Replay synchronized PiperX EEF and TRIMORPH leader-hand streams."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from collection.config import (
    FOLLOWER_PROFILE_ACCELERATION_RAD_S2,
    FOLLOWER_PROFILE_VELOCITY_RAD_S,
    RAW_EEF_FIELDS,
    apply_overrides,
    build_config,
    joint_command_scales_for_motor_ids,
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
REPLAY_HZ = 30.0
PREPOSITION_SECONDS = 3.0
SETTLE_SECONDS = 1.0
PIPER_PREPOSITION_SPEED_PERCENT = 20
PIPER_REPLAY_SPEED_PERCENT = 100
DEFAULT_THUMB_MODEL = Path("thumb_retarget_model.json")


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
) -> tuple[np.ndarray, np.ndarray]:
    """Retarget a MIDAS replay stream or preserve raw TRIMORPH positions."""

    if not use_follower:
        return stream
    times, positions = stream
    targets = np.asarray(
        [
            build_follower_target(
                position,
                motor_ids,
                command_scale=command_scale,
                joint_command_scales=joint_command_scales,
                thumb_model=thumb_model,
            )
            for position in positions
        ],
        dtype=np.float64,
    )
    return times, targets


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
    """Load either hand configuration narrowed to the recorded motor columns."""

    loader = load_follower_config if use_follower else load_hand_config
    return loader(hand_config, hand_port, active_motors)


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
    config = apply_overrides(
        build_config(), hand_config=args.hand_config, hand_port=args.hand_port
    )
    if hand_stream is not None:
        joint_scales = np.asarray(
            joint_command_scales_for_motor_ids(
                config.follower_joint_command_scales, active_motors or ()
            ),
            dtype=np.float64,
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
            thumb_model=thumb_model,
        )

    hand = None
    api = None
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
            hand.set_motion_profile(
                profile_velocity_rad_s=FOLLOWER_PROFILE_VELOCITY_RAD_S,
                profile_acceleration_rad_s2=FOLLOWER_PROFILE_ACCELERATION_RAD_S2,
            )
            hand.set_positions(hand.read_pos(), clip=True)
            hand.enable_torque()
            if args.use_follower:
                print(f"MIDAS hand connected on {hand.port}")
            else:
                print(f"TRIMORPH hand connected on {hand.port}")

        if replay_arm:
            from scripts.piper_api.piper_api import PiperAPI

            api = PiperAPI(
                firmware=config.piper_firmware,
                interface=config.piper_interface,
                channel=config.piper_channel,
                installation_pos=config.piper_installation_pos,
                payload=config.piper_payload,
            )
            api.connect()
            print("Piper connected:", api.is_ok(), "receive FPS:", api.get_fps())
            if not args.no_piper_enable and not api.enable():
                raise RuntimeError("Timed out while enabling Piper")

        move_to_first_pose(hand, api, hand_stream, piper_stream, start_s, args)
        if api is not None:
            api.set_speed_percent(PIPER_REPLAY_SPEED_PERCENT)

        period_s = 1.0 / args.rate_hz
        wall_start = time.monotonic()
        tick = 0
        print(
            f"Replaying {episode_dir} from {start_s:.3f}s to {end_s:.3f}s "
            f"at {args.rate_hz:.1f} Hz."
        )
        while True:
            episode_t = min(start_s + tick * period_s, end_s)
            if hand is not None and hand_stream is not None:
                hand.set_positions(sample_stream(*hand_stream, episode_t), clip=True)
            if api is not None and piper_stream is not None:
                api.move_eef(sample_stream(*piper_stream, episode_t))
            if episode_t >= end_s:
                break
            tick += 1
            sleep_s = wall_start + tick * period_s - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
        print("Replay complete.")
        return 0
    except KeyboardInterrupt:
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
