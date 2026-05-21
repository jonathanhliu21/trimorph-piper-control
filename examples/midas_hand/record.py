import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.midas_api import add_hand_connection_args, connect_hand  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Record Midas hand motor states to CSV.")
    add_hand_connection_args(parser)
    parser.add_argument(
        "--output",
        default="hand_recording.csv",
        help="CSV file to write. Default: hand_recording.csv",
    )
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to record. Use 0 until Ctrl+C. Default: 0",
    )
    parser.add_argument(
        "--enable-torque",
        action="store_true",
        help="Leave torque enabled while recording. Default disables torque.",
    )
    return parser.parse_args()


def fieldnames(motor_ids):
    fields = ["elapsed_s", "wall_time_s"]
    fields.extend(f"pos_{motor_id}" for motor_id in motor_ids)
    fields.extend(f"vel_{motor_id}" for motor_id in motor_ids)
    fields.extend(f"cur_{motor_id}" for motor_id in motor_ids)
    return fields


def main():
    args = parse_args()
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be greater than 0")
    if args.duration < 0:
        raise ValueError("--duration must be 0 or greater")

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hand = connect_hand(args, enable_torque=args.enable_torque)
    period_s = 1.0 / args.rate_hz
    start_t = time.monotonic()
    next_sample_t = start_t
    rows_written = 0

    try:
        print(f"Hand connected on {hand.port}")
        print(f"Recording to {output_path}. Press Ctrl+C to stop.")
        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames(hand.motor_ids))
            writer.writeheader()

            while True:
                now = time.monotonic()
                if args.duration and now - start_t >= args.duration:
                    break

                sleep_s = next_sample_t - now
                if sleep_s > 0:
                    time.sleep(sleep_s)

                pos, vel, cur = hand.read_pos_vel_cur()
                sample_t = time.monotonic()
                next_sample_t += period_s

                row = {
                    "elapsed_s": f"{sample_t - start_t:.6f}",
                    "wall_time_s": f"{time.time():.6f}",
                }
                for motor_id, value in zip(hand.motor_ids, pos):
                    row[f"pos_{motor_id}"] = f"{value:.9f}"
                for motor_id, value in zip(hand.motor_ids, vel):
                    row[f"vel_{motor_id}"] = f"{value:.9f}"
                for motor_id, value in zip(hand.motor_ids, cur):
                    row[f"cur_{motor_id}"] = f"{value:.9f}"

                writer.writerow(row)
                rows_written += 1
                if rows_written % max(1, int(args.rate_hz)) == 0:
                    csv_file.flush()
                    print(f"Recorded {rows_written} samples")

    except KeyboardInterrupt:
        print("\nRecording stopped.")
    finally:
        hand.shutdown()


if __name__ == "__main__":
    main()
