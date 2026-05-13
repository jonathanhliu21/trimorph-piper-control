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

JOINT_FIELDS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay Piper joint positions from a recorded CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="recorded_motion.csv",
        help="CSV file to replay. Default: recorded_motion.csv",
    )
    parser.add_argument(
        "--ready-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait before moving to the first pose. Default: 0",
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
        "--speed-percent",
        type=int,
        default=100,
        help="Arm speed percent sent before replay. Default: 100",
    )
    parser.add_argument(
        "--preposition-speed-percent",
        type=int,
        default=20,
        help="Arm speed percent used to move to the first recorded pose. Default: 20",
    )
    parser.add_argument(
        "--preposition-seconds",
        type=float,
        default=3.0,
        help="Seconds to allow for the slow move to the first recorded pose. Default: 3",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="Seconds to pause after reaching the first pose before replay. Default: 2",
    )
    parser.add_argument(
        "--no-enable",
        action="store_true",
        help="Do not send enable before replay.",
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


def read_motion(csv_path):
    rows = []
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = {"elapsed_s", *JOINT_FIELDS} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                elapsed_s = float(row["elapsed_s"])
                joints = [float(row[field]) for field in JOINT_FIELDS]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line_number}") from exc
            rows.append((elapsed_s, joints))

    rows.sort(key=lambda item: item[0])
    return rows


def wait_until_ready(seconds):
    if seconds <= 0:
        return
    print(f"Get ready. Initial positioning starts in {seconds:.1f} seconds.")
    time.sleep(seconds)


def enable_robot(robot, timeout_s=5.0):
    start_t = time.monotonic()
    while time.monotonic() - start_t < timeout_s:
        if robot.enable():
            return True
        time.sleep(0.05)
    return False


def move_to_initial_pose(robot, target_joints, args):
    print("Moving slowly to the first recorded pose...")
    robot.set_speed_percent(args.preposition_speed_percent)
    robot.move_j(target_joints)

    if args.preposition_seconds > 0:
        print(f"Allowing {args.preposition_seconds:.1f} seconds for initial positioning.")
        time.sleep(args.preposition_seconds)

    if args.settle_seconds > 0:
        print(f"Settling for {args.settle_seconds:.1f} seconds.")
        time.sleep(args.settle_seconds)


def main():
    args = parse_args()
    if args.ready_seconds < 0:
        raise ValueError("--ready-seconds must be 0 or greater")
    if not 0 <= args.speed_percent <= 100:
        raise ValueError("--speed-percent must be between 0 and 100")
    if not 0 <= args.preposition_speed_percent <= 100:
        raise ValueError("--preposition-speed-percent must be between 0 and 100")
    if args.preposition_seconds < 0:
        raise ValueError("--preposition-seconds must be 0 or greater")
    if args.settle_seconds < 0:
        raise ValueError("--settle-seconds must be 0 or greater")

    csv_path = Path(args.csv_path).expanduser()
    rows = read_motion(csv_path)
    if not rows:
        raise ValueError(f"No motion samples found in {csv_path}")

    robot = create_robot(args)
    robot.connect()

    try:
        time.sleep(0.5)
        print("Connected OK:", robot.is_ok())
        print("Receive FPS:", robot.get_fps())

        robot.set_speed_percent(args.speed_percent)
        if not args.no_enable:
            print("Enabling robot...")
            if not enable_robot(robot):
                raise RuntimeError("Timed out while enabling robot")

        wait_until_ready(args.ready_seconds)
        move_to_initial_pose(robot, rows[0][1], args)

        robot.set_speed_percent(args.speed_percent)
        first_elapsed_s = rows[0][0]
        replay_start_t = time.monotonic()
        print(f"Replaying {len(rows)} samples from {csv_path}")

        for elapsed_s, joints in rows:
            target_t = replay_start_t + (elapsed_s - first_elapsed_s)
            sleep_s = target_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            robot.move_j(joints)

        print("Replay complete.")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
