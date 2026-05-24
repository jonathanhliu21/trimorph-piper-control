#!/usr/bin/env python3
"""Replay a collected raw episode on the Piper arm and Midas hand."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from collection.config import (
    CANONICAL_MIDAS_MOTOR_IDS,
    RAW_EEF_FIELDS,
    build_config,
)
from collection.episode import raw_root
from collection.midas_worker import apply_midas_runtime_limits, load_midas_config


# Defaults. CLI flags can override the episode path and replay rate.
DATASET_ROOT = Path("data_midas")
REPLAY_HZ = 30.0
PREPOSITION = True
PREPOSITION_SECONDS = 3.0
SETTLE_SECONDS = 1.0
PIPER_ENABLE = True
PIPER_PREPOSITION_SPEED_PERCENT = 20
PIPER_REPLAY_SPEED_PERCENT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one raw data-collection episode. Defaults to the latest "
            "episode under data_midas/raw."
        )
    )
    parser.add_argument(
        "episode",
        nargs="?",
        default=None,
        help=(
            "Episode directory, episode index such as 42, or name such as "
            "episode_000042. Default: latest raw episode."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DATASET_ROOT),
        help="Dataset root containing raw episodes. Default: data_midas",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=REPLAY_HZ,
        help="Replay command frequency in Hz. Default: 30",
    )
    parser.add_argument(
        "--no-preposition",
        action="store_true",
        help="Skip moving slowly to the first recorded arm/hand pose.",
    )
    parser.add_argument(
        "--preposition-seconds",
        type=float,
        default=PREPOSITION_SECONDS,
        help="Seconds to resend the first pose before replay. Default: 3",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=SETTLE_SECONDS,
        help="Seconds to pause after prepositioning. Default: 1",
    )
    parser.add_argument(
        "--hand-only",
        action="store_true",
        help="Replay only Midas hand positions.",
    )
    parser.add_argument(
        "--arm-only",
        action="store_true",
        help="Replay only Piper EEF poses.",
    )
    parser.add_argument(
        "--no-piper-enable",
        action="store_true",
        help="Do not call Piper enable() before replay.",
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
    if candidate.exists():
        return candidate
    if episode.isdigit():
        candidate = root / f"episode_{int(episode):06d}"
    elif episode.startswith("episode_"):
        candidate = root / episode
    else:
        candidate = root / f"episode_{episode}"
    if not candidate.exists():
        raise FileNotFoundError(f"Episode not found: {candidate}")
    return candidate


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def load_hand_stream(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No hand samples in {path}")
    times = np.asarray([float(row["elapsed_s"]) for row in rows], dtype=np.float64)
    values = np.asarray(
        [
            [float(row[f"pos_{motor_id}"]) for motor_id in CANONICAL_MIDAS_MOTOR_IDS]
            for row in rows
        ],
        dtype=np.float64,
    )
    order = np.argsort(times)
    return times[order], values[order]


def load_piper_stream(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No Piper samples in {path}")
    times = np.asarray([float(row["elapsed_s"]) for row in rows], dtype=np.float64)
    values = np.asarray(
        [[float(row[field]) for field in RAW_EEF_FIELDS] for row in rows],
        dtype=np.float64,
    )
    order = np.argsort(times)
    return times[order], values[order]


def sample_stream(times: np.ndarray, values: np.ndarray, query_t: float) -> np.ndarray:
    return np.asarray(
        [np.interp(query_t, times, values[:, dim]) for dim in range(values.shape[1])],
        dtype=np.float64,
    )


def load_streams(
    episode_dir: Path,
    *,
    replay_hand: bool,
    replay_arm: bool,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, tuple[np.ndarray, np.ndarray] | None]:
    hand_stream = load_hand_stream(episode_dir / "midas.csv") if replay_hand else None
    piper_stream = load_piper_stream(episode_dir / "piper.csv") if replay_arm else None
    return hand_stream, piper_stream


def replay_duration(
    hand_stream: tuple[np.ndarray, np.ndarray] | None,
    piper_stream: tuple[np.ndarray, np.ndarray] | None,
) -> float:
    ends = []
    if hand_stream is not None:
        ends.append(float(hand_stream[0][-1]))
    if piper_stream is not None:
        ends.append(float(piper_stream[0][-1]))
    if not ends:
        raise ValueError("Nothing selected for replay")
    return min(ends)


def move_to_first_pose(hand, api, hand_stream, piper_stream, args) -> None:
    if args.no_preposition:
        return
    print("Moving to first recorded pose...")
    if api is not None:
        api.set_speed_percent(PIPER_PREPOSITION_SPEED_PERCENT)
    first_hand = hand_stream[1][0] if hand_stream is not None else None
    first_eef = piper_stream[1][0] if piper_stream is not None else None

    end_t = time.monotonic() + args.preposition_seconds
    while time.monotonic() < end_t:
        if hand is not None and first_hand is not None:
            hand.set_positions(first_hand, clip=True)
        if api is not None and first_eef is not None:
            api.move_eef(first_eef)
        time.sleep(0.05)

    if args.settle_seconds > 0:
        print(f"Settling for {args.settle_seconds:.1f}s.")
        time.sleep(args.settle_seconds)


def main() -> int:
    args = parse_args()
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be greater than 0")
    if args.preposition_seconds < 0:
        raise ValueError("--preposition-seconds must be 0 or greater")
    if args.settle_seconds < 0:
        raise ValueError("--settle-seconds must be 0 or greater")
    if args.hand_only and args.arm_only:
        raise ValueError("--hand-only and --arm-only are mutually exclusive")

    replay_hand = not args.arm_only
    replay_arm = not args.hand_only
    dataset_root = Path(args.dataset_root).expanduser()
    episode_dir = resolve_episode(dataset_root, args.episode)
    hand_stream, piper_stream = load_streams(
        episode_dir,
        replay_hand=replay_hand,
        replay_arm=replay_arm,
    )
    duration_s = replay_duration(hand_stream, piper_stream)

    config = build_config()
    hand = None
    api = None
    try:
        if replay_hand:
            from midas_hand_api import MidasHand

            hand = MidasHand(config=load_midas_config(config), autoconnect=True)
            hand.configure(enable_torque=False)
            apply_midas_runtime_limits(hand, config)
            hand.enable_torque()
            if [int(motor_id) for motor_id in hand.motor_ids] != list(
                CANONICAL_MIDAS_MOTOR_IDS
            ):
                raise RuntimeError(
                    f"Connected Midas motor IDs {hand.motor_ids} do not match recording order"
                )
            print(f"Midas hand connected on {hand.port}")

        if replay_arm:
            from scripts.piper_api.piper_api import PiperAPI

            api = PiperAPI(
                firmware=config.piper_firmware,
                interface=config.piper_interface,
                channel=config.piper_channel,
                installation_pos=config.piper_installation_pos,
            )
            api.connect()
            print("Piper connected:", api.is_ok(), "receive FPS:", api.get_fps())
            if PIPER_ENABLE and not args.no_piper_enable:
                print("Enabling Piper...")
                if not api.enable():
                    raise RuntimeError("Timed out while enabling Piper")

        move_to_first_pose(hand, api, hand_stream, piper_stream, args)
        if api is not None:
            api.set_speed_percent(PIPER_REPLAY_SPEED_PERCENT)

        period_s = 1.0 / args.rate_hz
        start_t = time.monotonic()
        next_tick_t = start_t
        tick = 0
        print(
            f"Replaying {episode_dir} for {duration_s:.3f}s at {args.rate_hz:.1f}Hz "
            f"({'hand' if replay_hand else ''}{'+' if replay_hand and replay_arm else ''}"
            f"{'arm' if replay_arm else ''})."
        )
        while True:
            elapsed_s = time.monotonic() - start_t
            if elapsed_s > duration_s:
                break
            if hand is not None and hand_stream is not None:
                hand.set_positions(sample_stream(*hand_stream, elapsed_s), clip=True)
            if api is not None and piper_stream is not None:
                api.move_eef(sample_stream(*piper_stream, elapsed_s))

            tick += 1
            next_tick_t = start_t + tick * period_s
            sleep_s = next_tick_t - time.monotonic()
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
