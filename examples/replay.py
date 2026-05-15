import argparse
import csv
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.piper_api import PiperAPI  # noqa: E402


JOINT_FIELDS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
FLANGE_FIELDS = ["x", "y", "z", "roll", "pitch", "yaw"]
FIRMWARE_STATUS_FIELDS = [
    "ctrl_mode",
    "arm_status",
    "mode_feedback",
    "teach_status",
    "motion_status",
    "trajectory_num",
    "err_code",
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
        choices=("default", "v183", "v188"),
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
        "--preposition-command-hz",
        type=float,
        default=10.0,
        help="Rate used to resend the first recorded pose during prepositioning. Default: 10",
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
    parser.add_argument(
        "--control-mode",
        choices=("cpv", "ik", "move-js", "move-j"),
        default="move-js",
        help="Control API used during replay. Default: move-js",
    )
    parser.add_argument(
        "--status-log-interval",
        type=float,
        default=0.5,
        help=(
            "Seconds between firmware status checks during replay. "
            "Use 0 to disable. Default: 0.5"
        ),
    )
    return parser.parse_args()


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
                flange_pose = None
                if all(row.get(field) not in (None, "") for field in FLANGE_FIELDS):
                    flange_pose = [float(row[field]) for field in FLANGE_FIELDS]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line_number}") from exc
            rows.append((elapsed_s, joints, flange_pose))

    rows.sort(key=lambda item: item[0])
    return rows


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


def wait_for_not_teacher_mode(api, *, poll_s=0.25, stable_s=0.5):
    print("Checking teacher mode before replay...")
    next_notice_t = 0.0
    inactive_since_t = None
    while True:
        is_teacher_mode, status = get_teacher_mode(api)
        now = time.monotonic()
        if not is_teacher_mode:
            if inactive_since_t is None:
                inactive_since_t = now
            if now - inactive_since_t >= stable_s:
                print("Teacher mode is inactive.")
                return
        else:
            inactive_since_t = None

        if now >= next_notice_t:
            if is_teacher_mode:
                print(
                    "Teacher mode is active. "
                    f"Current teach_status={status!r}. "
                    "Leave teacher mode to start replay."
                )
            else:
                print(
                    "Teacher mode is inactive. "
                    f"Current teach_status={status!r}. "
                    f"Waiting {stable_s:.1f} seconds for it to stay inactive."
                )
            next_notice_t = now + 2.0
        time.sleep(poll_s)


def wait_until_ready(seconds):
    if seconds <= 0:
        return
    print(f"Get ready. Initial positioning starts in {seconds:.1f} seconds.")
    time.sleep(seconds)


def collect_firmware_status(api):
    state = api.get_state()
    return {
        field: "" if state.get(field) is None else str(state.get(field))
        for field in FIRMWARE_STATUS_FIELDS
    }


def format_firmware_status(status, previous_status=None):
    parts = []
    for field in FIRMWARE_STATUS_FIELDS:
        value = status[field]
        if previous_status is not None and value == previous_status.get(field):
            continue
        parts.append(f"{field}={value!r}")
    return ", ".join(parts)


def maybe_log_firmware_status(
    api,
    previous_status,
    next_check_t,
    *,
    interval_s,
    label,
    force=False,
):
    if interval_s <= 0 and not force:
        return previous_status, next_check_t

    now = time.monotonic()
    if not force and now < next_check_t:
        return previous_status, next_check_t

    status = collect_firmware_status(api)
    if force or status != previous_status:
        formatted_status = format_firmware_status(
            status,
            None if force else previous_status,
        )
        if formatted_status:
            print(f"{label} firmware status: {formatted_status}")

    return status, now + interval_s


def validate_motion_for_control_mode(rows, control_mode):
    if control_mode != "ik":
        return

    missing_pose_rows = [
        row_number
        for row_number, row in enumerate(rows, start=2)
        if row[2] is None
    ]
    if missing_pose_rows:
        preview = ", ".join(str(row_number) for row_number in missing_pose_rows[:5])
        if len(missing_pose_rows) > 5:
            preview = f"{preview}, ..."
        raise ValueError(
            "IK replay requires complete x, y, z, roll, pitch, yaw columns "
            f"on every CSV row. Missing pose data on row(s): {preview}"
        )


def move_to_initial_pose(api, target_joints, args, status_log):
    print("Moving slowly to the first recorded pose...")
    api.set_speed_percent(args.preposition_speed_percent)
    api.move_joint(target_joints, mode="smooth")
    status_log["status"], status_log["next_check_t"] = maybe_log_firmware_status(
        api,
        status_log["status"],
        status_log["next_check_t"],
        interval_s=args.status_log_interval,
        label="Preposition",
        force=True,
    )

    if args.preposition_seconds > 0:
        print(
            f"Prepositioning for {args.preposition_seconds:.1f} seconds "
            f"at {args.preposition_speed_percent}% speed."
        )
        end_t = time.monotonic() + args.preposition_seconds
        command_period_s = 1.0 / args.preposition_command_hz
        while True:
            sleep_s = min(command_period_s, end_t - time.monotonic())
            if sleep_s <= 0:
                break
            time.sleep(sleep_s)
            api.move_joint(target_joints, mode="smooth")
            status_log["status"], status_log["next_check_t"] = (
                maybe_log_firmware_status(
                    api,
                    status_log["status"],
                    status_log["next_check_t"],
                    interval_s=args.status_log_interval,
                    label="Preposition",
                )
            )

    if args.settle_seconds > 0:
        print(f"Settling for {args.settle_seconds:.1f} seconds.")
        time.sleep(args.settle_seconds)


def command_replay_target(api, target_joints, target_flange_pose, control_mode):
    if control_mode == "cpv":
        api.move_joint(target_joints, mode="cpv")
    elif control_mode == "ik":
        if target_flange_pose is None:
            raise ValueError("IK replay requires x, y, z, roll, pitch, yaw columns")
        api.move_eef(target_flange_pose)
    elif control_mode == "move-js":
        api.move_joint(target_joints, mode="exact")
    elif control_mode == "move-j":
        api.move_joint(target_joints, mode="smooth")
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}")


def run_replay(api, rows, csv_path, args, status_log):
    first_elapsed_s = rows[0][0]
    replay_start_t = time.monotonic()
    sample_index = 0

    print(f"Replaying {len(rows)} samples from {csv_path} with {args.control_mode}")
    status_log["status"], status_log["next_check_t"] = maybe_log_firmware_status(
        api,
        status_log["status"],
        status_log["next_check_t"],
        interval_s=args.status_log_interval,
        label="Replay",
        force=True,
    )

    while sample_index < len(rows):
        now = time.monotonic()
        replay_elapsed_s = first_elapsed_s + (now - replay_start_t)
        status_log["status"], status_log["next_check_t"] = maybe_log_firmware_status(
            api,
            status_log["status"],
            status_log["next_check_t"],
            interval_s=args.status_log_interval,
            label=f"Replay t={replay_elapsed_s:.3f}s",
        )

        while sample_index < len(rows) and rows[sample_index][0] <= replay_elapsed_s:
            target_joints = rows[sample_index][1]
            target_flange_pose = rows[sample_index][2]
            command_replay_target(
                api,
                target_joints,
                target_flange_pose,
                args.control_mode,
            )
            sample_index += 1

        if sample_index < len(rows):
            next_wake_t = replay_start_t + (rows[sample_index][0] - first_elapsed_s)
            sleep_s = max(0.001, min(0.02, next_wake_t - time.monotonic()))
        else:
            sleep_s = 0.001
        time.sleep(sleep_s)


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
    if args.preposition_command_hz <= 0:
        raise ValueError("--preposition-command-hz must be greater than 0")
    if args.settle_seconds < 0:
        raise ValueError("--settle-seconds must be 0 or greater")
    if args.status_log_interval < 0:
        raise ValueError("--status-log-interval must be 0 or greater")
    csv_path = Path(args.csv_path).expanduser()
    rows = read_motion(csv_path)
    if not rows:
        raise ValueError(f"No motion samples found in {csv_path}")
    validate_motion_for_control_mode(rows, args.control_mode)

    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
    )
    api.connect()

    try:
        print("Connected OK:", api.is_ok())
        print("Receive FPS:", api.get_fps())

        wait_for_not_teacher_mode(api)
        wait_until_ready(args.ready_seconds)

        api.set_speed_percent(args.preposition_speed_percent)
        if not args.no_enable:
            print("Enabling robot...")
            if not api.enable():
                raise RuntimeError("Timed out while enabling robot")

        status_log = {"status": None, "next_check_t": 0.0}
        move_to_initial_pose(api, rows[0][1], args, status_log)

        api.set_speed_percent(args.speed_percent)
        run_replay(api, rows, csv_path, args, status_log)

        print("Replay complete.")

    finally:
        api.disconnect()


if __name__ == "__main__":
    main()
