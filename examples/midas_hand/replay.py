import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.midas_api import add_hand_connection_args, connect_hand  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Replay a Midas hand recording CSV.")
    add_hand_connection_args(parser)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="hand_recording.csv",
        help="CSV file to replay. Default: hand_recording.csv",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. Default: 1",
    )
    parser.add_argument(
        "--preposition",
        action="store_true",
        help="Move to the first recorded pose before timed replay.",
    )
    parser.add_argument("--preposition-timeout", type=float, default=5.0)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--profile-velocity", type=float, default=None)
    parser.add_argument("--profile-acceleration", type=int, default=None)
    return parser.parse_args()


def read_recording(csv_path):
    rows = []
    with Path(csv_path).expanduser().open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if "elapsed_s" not in (reader.fieldnames or []):
            raise ValueError("CSV is missing elapsed_s")
        motor_ids = [
            int(field.removeprefix("pos_"))
            for field in reader.fieldnames or []
            if field.startswith("pos_")
        ]
        if not motor_ids:
            raise ValueError("CSV has no pos_<motor_id> columns")

        for line_number, row in enumerate(reader, start=2):
            try:
                elapsed_s = float(row["elapsed_s"])
                positions = np.asarray(
                    [float(row[f"pos_{motor_id}"]) for motor_id in motor_ids],
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line_number}") from exc
            rows.append((elapsed_s, positions))

    rows.sort(key=lambda item: item[0])
    return motor_ids, rows


def main():
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be greater than 0")
    if args.preposition_timeout < 0:
        raise ValueError("--preposition-timeout must be 0 or greater")
    if args.tolerance < 0:
        raise ValueError("--tolerance must be 0 or greater")

    motor_ids, rows = read_recording(args.csv_path)
    if not rows:
        raise ValueError(f"No hand samples found in {args.csv_path}")

    hand = connect_hand(args, enable_torque=True)
    try:
        if list(hand.motor_ids) != motor_ids:
            raise ValueError(
                f"Recording motor IDs {motor_ids} do not match connected config {hand.motor_ids}"
            )
        if args.profile_velocity is not None or args.profile_acceleration is not None:
            hand.set_motion_profile(
                profile_velocity_rad_s=args.profile_velocity,
                profile_acceleration_raw=args.profile_acceleration,
            )

        print(f"Hand connected on {hand.port}")
        if args.preposition:
            print("Moving to first recorded pose...")
            hand.set_positions_blocking(
                rows[0][1],
                tolerance_rad=args.tolerance,
                timeout_s=args.preposition_timeout,
            )

        print(f"Replaying {len(rows)} hand samples from {args.csv_path}")
        first_elapsed_s = rows[0][0]
        replay_start_t = time.monotonic()
        sample_index = 0
        while sample_index < len(rows):
            now = time.monotonic()
            replay_elapsed_s = first_elapsed_s + (now - replay_start_t) * args.speed

            while sample_index < len(rows) and rows[sample_index][0] <= replay_elapsed_s:
                hand.set_positions(rows[sample_index][1], clip=True)
                sample_index += 1

            if sample_index < len(rows):
                next_wake_t = replay_start_t + (
                    (rows[sample_index][0] - first_elapsed_s) / args.speed
                )
                sleep_s = max(0.001, min(0.02, next_wake_t - time.monotonic()))
                time.sleep(sleep_s)

        print("Replay complete.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        hand.shutdown()


if __name__ == "__main__":
    main()
