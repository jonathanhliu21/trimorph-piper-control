import argparse
import csv
import time
from pathlib import Path

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config


FIRMWARE = {
    "default": PiperFW.DEFAULT,
    "v183": PiperFW.V183,
    "v188": PiperFW.V188,
}

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
        default=60.0,
        help="Sampling rate in Hz. Default: 60",
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
        choices=sorted(FIRMWARE),
        default="default",
        help="Piper firmware driver selection. Default: default",
    )
    parser.add_argument(
        "--keep-torque-on",
        action="store_true",
        help="Do not disable joint motors before recording.",
    )
    return parser.parse_args()


def create_robot(args):
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=FIRMWARE[args.firmware],
        interface=args.interface,
        channel=args.channel,
    )
    return AgxArmFactory.create_arm(cfg)


def wait_until_ready(seconds):
    if seconds <= 0:
        return
    print(f"Get ready. Recording starts in {seconds:.1f} seconds.")
    time.sleep(seconds)


def disable_robot(robot, attempts=5):
    for _ in range(attempts):
        if robot.disable():
            return True
        time.sleep(0.1)
    return False


def main():
    args = parse_args()
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be greater than 0")
    if args.duration < 0:
        raise ValueError("--duration must be 0 or greater")

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    period_s = 1.0 / args.rate_hz

    robot = create_robot(args)
    robot.connect()

    try:
        time.sleep(0.5)
        print("Connected OK:", robot.is_ok())
        print("Receive FPS:", robot.get_fps())

        if not args.keep_torque_on:
            print("Disabling all joint motors for recording...")
            if disable_robot(robot):
                print("Joint motors disabled.")
            else:
                raise RuntimeError("Failed to disable joint motors before recording")

        wait_until_ready(args.ready_seconds)

        rows_written = 0
        start_t = time.monotonic()
        next_sample_t = start_t

        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()

            print(f"Recording to {output_path}. Press Ctrl+C to stop.")
            while True:
                now = time.monotonic()
                if args.duration and now - start_t >= args.duration:
                    break

                sleep_s = next_sample_t - now
                if sleep_s > 0:
                    time.sleep(sleep_s)

                joints = robot.get_joint_angles()
                flange = robot.get_flange_pose()
                sample_t = time.monotonic()
                next_sample_t += period_s

                if joints is None:
                    continue

                row = {
                    "elapsed_s": f"{sample_t - start_t:.6f}",
                    "robot_timestamp": f"{joints.timestamp:.6f}",
                }
                row.update(
                    {f"joint_{i + 1}": f"{value:.9f}" for i, value in enumerate(joints.msg)}
                )

                if flange is not None:
                    row.update(
                        {
                            name: f"{value:.9f}"
                            for name, value in zip(
                                ["x", "y", "z", "roll", "pitch", "yaw"], flange.msg
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
        robot.disconnect()


if __name__ == "__main__":
    main()
