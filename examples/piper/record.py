import argparse
import csv
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.piper_api.foot_pedal import (  # noqa: E402
    FootPedalTrigger,
    format_input_devices,
    list_input_devices,
)


FIELDNAMES = [
    "elapsed_s",
    "robot_timestamp",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
]

TEACHER_MODE_ACTIVE_VALUES = {
    "1",
    "true",
    "teach",
    "teacher",
    "teaching",
    "start_recording",
    "start recording",
    "start_recording(0x1)",
    "0x1",
    "enabled",
    "on",
}
TEACHER_MODE_INACTIVE_VALUES = {
    "",
    "0",
    "false",
    "none",
    "normal",
    "disabled",
    "off",
    "stop_recording",
    "stop recording",
    "stop_recording(0x2)",
    "0x2",
    "not teach",
    "not_teacher",
    "not teacher",
    "no_teach",
    "no teach",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record Piper joint positions and flange pose to CSV."
    )
    parser.add_argument(
        "--output",
        default="recorded_motion.csv",
        help="CSV file to write. Default: recorded_motion.csv",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=100.0,
        help="Sampling rate in Hz. Default: 100",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Recording duration in seconds. Use 0 to record until Ctrl+C. Default: 0",
    )
    parser.add_argument(
        "--ready-seconds",
        type=float,
        default=3.0,
        help="Seconds to wait before recording starts. Default: 3",
    )
    parser.add_argument(
        "--interface",
        default="socketcan",
        help="python-can interface. Default: socketcan",
    )
    parser.add_argument(
        "--channel",
        default="can0",
        help="CAN channel. Default: can0",
    )
    parser.add_argument(
        "--firmware",
        choices=("default", "v183", "v188"),
        default="default",
        help="Piper firmware driver selection. Default: default",
    )
    parser.add_argument(
        "--foot-pedal",
        action="store_true",
        help=(
            "Use a Linux input foot pedal to start and stop recording. "
            "Auto-detects devices with foot/pedal in the name."
        ),
    )
    parser.add_argument(
        "--foot-pedal-device",
        default=None,
        help="Foot pedal event device, for example /dev/input/event7.",
    )
    parser.add_argument(
        "--foot-pedal-key-code",
        type=int,
        default=None,
        help="Only accept this Linux input key code from the pedal. Default: any key.",
    )
    parser.add_argument(
        "--foot-pedal-debounce-seconds",
        type=float,
        default=0.5,
        help="Ignore repeated pedal presses for this long. Default: 0.5",
    )
    parser.add_argument(
        "--list-input-devices",
        action="store_true",
        help="List /dev/input event devices and exit.",
    )
    return parser.parse_args()


def is_teacher_mode_status(value):
    normalized = str(value).strip().lower()
    if normalized in TEACHER_MODE_ACTIVE_VALUES:
        return True
    if normalized in TEACHER_MODE_INACTIVE_VALUES:
        return False
    if "start_recording" in normalized or "start recording" in normalized:
        return True
    if "stop_recording" in normalized or "stop recording" in normalized:
        return False
    return "teach" in normalized


def get_teacher_mode(api):
    status = api.get_state().get("teach_status")
    return is_teacher_mode_status(status), status


def wait_for_teacher_mode(api, *, poll_s=0.25):
    print("Checking teacher mode before recording...")
    next_notice_t = 0.0
    while True:
        is_teacher_mode, status = get_teacher_mode(api)
        if is_teacher_mode:
            print("Teacher mode is active.")
            return

        now = time.monotonic()
        if now >= next_notice_t:
            print(
                "Teacher mode is not active. "
                f"Current teach_status={status!r}. "
                "Put the robot in teacher mode to start recording."
            )
            next_notice_t = now + 2.0
        time.sleep(poll_s)


def wait_until_ready(seconds):
    if seconds <= 0:
        return
    print(f"Get ready. Recording starts in {seconds:.1f} seconds.")
    time.sleep(seconds)


def should_use_foot_pedal(args):
    return args.foot_pedal or args.foot_pedal_device is not None


def wait_for_foot_pedal(pedal, message, *, poll_s=0.01):
    print(message)
    while True:
        events = pedal.poll()
        if events:
            print(events[-1].message)
            return
        time.sleep(poll_s)


def main():
    args = parse_args()
    if args.list_input_devices:
        print(format_input_devices(list_input_devices()))
        return

    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be greater than 0")
    if args.duration < 0:
        raise ValueError("--duration must be 0 or greater")
    if args.foot_pedal_debounce_seconds < 0:
        raise ValueError("--foot-pedal-debounce-seconds must be 0 or greater")

    from scripts.piper_api.piper_api import PiperAPI

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    period_s = 1.0 / args.rate_hz

    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
    )
    api.connect()

    pedal = None
    try:
        print("Connected OK:", api.is_ok())
        print("Receive FPS:", api.get_fps())

        wait_for_teacher_mode(api)
        if should_use_foot_pedal(args):
            pedal = FootPedalTrigger(
                args.foot_pedal_device,
                key_code=args.foot_pedal_key_code,
                debounce_s=args.foot_pedal_debounce_seconds,
            )
            pedal.start()
            print(f"Foot pedal input: {pedal.active_device}")
            wait_for_foot_pedal(pedal, "Press the foot pedal to start recording.")
        wait_until_ready(args.ready_seconds)

        rows_written = 0
        start_t = time.monotonic()
        next_sample_t = start_t

        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()

            print(f"Recording to {output_path}. Press Ctrl+C to stop.")
            if pedal is not None:
                print("Press the foot pedal again to stop recording.")
            while True:
                now = time.monotonic()
                if args.duration and now - start_t >= args.duration:
                    break
                if pedal is not None and pedal.poll():
                    print("Foot pedal pressed; stopping recording.")
                    break

                sleep_s = next_sample_t - now
                if sleep_s > 0:
                    time.sleep(sleep_s)

                joints = api.get_joint_state()
                flange = api.get_eef_state()
                sample_t = time.monotonic()
                next_sample_t += period_s

                if joints is None:
                    continue

                row = {
                    "elapsed_s": f"{sample_t - start_t:.6f}",
                    "robot_timestamp": f"{joints.timestamp:.6f}",
                }
                row.update(
                    {
                        f"joint_{i + 1}": f"{value:.9f}"
                        for i, value in enumerate(joints.joints)
                    }
                )

                if flange is not None:
                    row.update(
                        {
                            name: f"{value:.9f}"
                            for name, value in zip(
                                ["x", "y", "z", "roll", "pitch", "yaw"],
                                flange.pose,
                            )
                        }
                    )
                else:
                    row.update({name: "" for name in ["x", "y", "z", "roll", "pitch", "yaw"]})

                writer.writerow(row)
                rows_written += 1

                if rows_written % max(1, int(args.rate_hz)) == 0:
                    csv_file.flush()
                    print(f"Recorded {rows_written} samples")

    except KeyboardInterrupt:
        print("\nRecording stopped.")
    finally:
        if pedal is not None:
            pedal.stop()
        api.disconnect()


if __name__ == "__main__":
    main()
