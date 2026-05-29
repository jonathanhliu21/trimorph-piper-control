import argparse
import csv
import select
import sys
import termios
import time
import tty
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.midas_api import add_hand_connection_args, connect_hand  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Record Midas hand motor states to CSV.")
    add_hand_connection_args(parser)
    parser.add_argument(
        "--csv",
        "--output",
        dest="csv_path",
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
    parser.add_argument(
        "--start-immediately",
        action="store_true",
        help="Start recording immediately instead of waiting for the r key.",
    )
    parser.add_argument(
        "--no-key-control",
        action="store_true",
        help="Disable r/q key controls and record until duration or Ctrl+C.",
    )
    return parser.parse_args()


def fieldnames(motor_ids):
    fields = ["elapsed_s", "wall_time_s"]
    fields.extend(f"pos_{motor_id}" for motor_id in motor_ids)
    fields.extend(f"vel_{motor_id}" for motor_id in motor_ids)
    fields.extend(f"cur_{motor_id}" for motor_id in motor_ids)
    return fields


class KeyboardReader:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._fd = None
        self._old_settings = None

    def __enter__(self):
        if self.enabled and sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self):
        if self._fd is None:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.read(1).lower()


def make_row(hand, start_t):
    pos, vel, cur = hand.read_pos_vel_cur()
    sample_t = time.monotonic()
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
    return row


def main():
    args = parse_args()
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be greater than 0")
    if args.duration < 0:
        raise ValueError("--duration must be 0 or greater")

    output_path = Path(args.csv_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hand = connect_hand(args, enable_torque=args.enable_torque)
    period_s = 1.0 / args.rate_hz
    key_control = not args.no_key_control and sys.stdin.isatty()
    recording = args.start_immediately or not key_control
    start_t = time.monotonic() if recording else None
    next_sample_t = start_t if recording else None
    rows_written = 0

    try:
        print(f"Hand connected on {hand.port}")
        if key_control:
            print(
                f"Recording to {output_path}. Press r to start/stop, q to quit, "
                "or Ctrl+C to stop."
            )
        else:
            print(f"Recording to {output_path}. Press Ctrl+C to stop.")
        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames(hand.motor_ids))
            writer.writeheader()

            with KeyboardReader(enabled=key_control) as keyboard:
                while True:
                    key = keyboard.read_key() if key_control else None
                    if key == "q":
                        break
                    if key == "r":
                        recording = not recording
                        if recording:
                            start_t = time.monotonic()
                            next_sample_t = start_t
                            rows_written = 0
                            print("\nRecording started.")
                        else:
                            csv_file.flush()
                            print(f"\nRecording stopped after {rows_written} samples.")
                            break
                        continue

                    if not recording:
                        time.sleep(0.02)
                        continue

                    assert start_t is not None
                    assert next_sample_t is not None

                    now = time.monotonic()
                    if args.duration and now - start_t >= args.duration:
                        if key_control:
                            recording = False
                            csv_file.flush()
                            print(
                                f"\nRecording stopped after {rows_written} samples "
                                "because duration was reached."
                            )
                            break
                        break

                    sleep_s = next_sample_t - now
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                    writer.writerow(make_row(hand, start_t))
                    rows_written += 1
                    next_sample_t += period_s
                    if rows_written % max(1, int(args.rate_hz)) == 0:
                        csv_file.flush()
                        print(f"Recorded {rows_written} samples")

    except KeyboardInterrupt:
        print("\nRecording stopped.")
    finally:
        hand.shutdown()


if __name__ == "__main__":
    main()
