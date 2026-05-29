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


THUMB_MOTOR_IDS = (0, 1, 2, 3)


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
        "--csv",
        dest="csv_arg",
        default=None,
        help="CSV file to replay. Overrides the positional CSV path.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. Default: 1",
    )
    preposition_group = parser.add_mutually_exclusive_group()
    preposition_group.add_argument(
        "--preposition",
        dest="preposition",
        action="store_true",
        help="Smoothly move to the first recorded pose before timed replay. Default.",
    )
    preposition_group.add_argument(
        "--no-preposition",
        dest="preposition",
        action="store_false",
        help="Start timed replay immediately from the current hand pose.",
    )
    parser.set_defaults(preposition=True)
    parser.add_argument("--preposition-duration", type=float, default=3.0)
    parser.add_argument("--preposition-rate-hz", type=float, default=60.0)
    parser.add_argument("--preposition-timeout", type=float, default=5.0)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--profile-velocity",
        type=float,
        default=2.5,
        help="Dynamixel profile velocity in rad/s. Default: 2.5.",
    )
    parser.add_argument(
        "--profile-acceleration",
        type=float,
        default=20.0,
        help="Dynamixel profile acceleration in rad/s^2. Default: 30.",
    )
    parser.add_argument("--p-gain", type=int, default=900, help="Replay position P gain. Default: 900.")
    parser.add_argument("--i-gain", type=int, default=None, help="Replay position I gain.")
    parser.add_argument("--d-gain", type=int, default=None, help="Replay position D gain.")
    parser.add_argument(
        "--thumb-current-rate-hz",
        type=float,
        default=5.0,
        help="Terminal print rate for thumb motor currents in mA. Use 0 to disable.",
    )
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


def thumb_motor_indices(hand):
    indices = []
    for motor_id in THUMB_MOTOR_IDS:
        if motor_id in hand.motor_ids:
            indices.append((motor_id, hand.motor_ids.index(motor_id)))
    return indices


def print_thumb_currents(hand, indices):
    currents = hand.read_cur()
    values = "  ".join(
        f"id{motor_id}: {currents[index]:7.1f} mA" for motor_id, index in indices
    )
    print(f"\rThumb currents | {values}", end="", flush=True)


def smooth_move_to_pose(
    hand,
    target,
    *,
    duration_s,
    rate_hz,
    tolerance_rad,
    settle_timeout_s,
):
    target = hand.clip_positions(target)
    current = hand.read_pos()

    if duration_s <= 0:
        hand.set_positions_blocking(
            target,
            tolerance_rad=tolerance_rad,
            timeout_s=settle_timeout_s,
        )
        return

    period_s = 1.0 / rate_hz
    steps = max(1, int(round(duration_s * rate_hz)))
    start_t = time.monotonic()
    for step in range(1, steps + 1):
        alpha = step / steps
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        command = current + (target - current) * smooth_alpha
        hand.set_positions(command, clip=True)

        next_t = start_t + step * period_s
        sleep_s = next_t - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)

    hand.set_positions(target, clip=True)
    if settle_timeout_s > 0:
        reached = hand.wait_until_reached(
            target,
            tolerance_rad=tolerance_rad,
            timeout_s=settle_timeout_s,
        )
        if not reached:
            raise TimeoutError(
                f"Motors did not settle at the first recorded pose within "
                f"{settle_timeout_s:.2f}s"
            )


def main():
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be greater than 0")
    if args.preposition_duration < 0:
        raise ValueError("--preposition-duration must be 0 or greater")
    if args.preposition_rate_hz <= 0:
        raise ValueError("--preposition-rate-hz must be greater than 0")
    if args.preposition_timeout < 0:
        raise ValueError("--preposition-timeout must be 0 or greater")
    if args.tolerance < 0:
        raise ValueError("--tolerance must be 0 or greater")
    if args.thumb_current_rate_hz < 0:
        raise ValueError("--thumb-current-rate-hz must be 0 or greater")
    for gain_name in ("p_gain", "i_gain", "d_gain"):
        gain = getattr(args, gain_name)
        if gain is not None and gain < 0:
            raise ValueError(f"--{gain_name.replace('_', '-')} must be non-negative")

    csv_path = args.csv_arg or args.csv_path
    motor_ids, rows = read_recording(csv_path)
    if not rows:
        raise ValueError(f"No hand samples found in {csv_path}")

    hand = connect_hand(args, enable_torque=True)
    try:
        if list(hand.motor_ids) != motor_ids:
            raise ValueError(
                f"Recording motor IDs {motor_ids} do not match connected config {hand.motor_ids}"
            )
        if (
            args.p_gain is not None
            or args.i_gain is not None
            or args.d_gain is not None
        ):
            hand.set_gains(p=args.p_gain, i=args.i_gain, d=args.d_gain)
            print(
                "Applied replay PID gains: "
                f"P={args.p_gain if args.p_gain is not None else 'unchanged'}, "
                f"I={args.i_gain if args.i_gain is not None else 'unchanged'}, "
                f"D={args.d_gain if args.d_gain is not None else 'unchanged'}"
            )
        if args.profile_velocity is not None or args.profile_acceleration is not None:
            hand.set_motion_profile(
                profile_velocity_rad_s=args.profile_velocity,
                profile_acceleration_rad_s2=args.profile_acceleration,
            )

        print(f"Hand connected on {hand.port}")
        if args.preposition:
            print(
                "Smoothly moving from current pose to first recorded pose "
                f"over {args.preposition_duration:.2f}s..."
            )
            smooth_move_to_pose(
                hand,
                rows[0][1],
                duration_s=args.preposition_duration,
                rate_hz=args.preposition_rate_hz,
                tolerance_rad=args.tolerance,
                settle_timeout_s=args.preposition_timeout,
            )

        print(f"Replaying {len(rows)} hand samples from {csv_path}")
        thumb_indices = thumb_motor_indices(hand)
        if args.thumb_current_rate_hz > 0 and thumb_indices:
            thumb_ids = ", ".join(str(motor_id) for motor_id, _ in thumb_indices)
            print(f"Printing thumb motor currents for IDs: {thumb_ids}")
        elif args.thumb_current_rate_hz > 0:
            print("No thumb motor IDs 0-3 are present in this hand config.")

        first_elapsed_s = rows[0][0]
        replay_start_t = time.monotonic()
        current_print_period_s = (
            1.0 / args.thumb_current_rate_hz
            if args.thumb_current_rate_hz > 0 and thumb_indices
            else None
        )
        next_current_print_t = replay_start_t
        sample_index = 0
        while sample_index < len(rows):
            now = time.monotonic()
            replay_elapsed_s = first_elapsed_s + (now - replay_start_t) * args.speed

            while sample_index < len(rows) and rows[sample_index][0] <= replay_elapsed_s:
                hand.set_positions(rows[sample_index][1], clip=True)
                sample_index += 1

            if (
                current_print_period_s is not None
                and time.monotonic() >= next_current_print_t
            ):
                print_thumb_currents(hand, thumb_indices)
                next_current_print_t += current_print_period_s

            if sample_index < len(rows):
                next_wake_t = replay_start_t + (
                    (rows[sample_index][0] - first_elapsed_s) / args.speed
                )
                sleep_s = max(0.001, min(0.02, next_wake_t - time.monotonic()))
                time.sleep(sleep_s)

        if current_print_period_s is not None:
            print()
        print("Replay complete.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        hand.shutdown()


if __name__ == "__main__":
    main()
