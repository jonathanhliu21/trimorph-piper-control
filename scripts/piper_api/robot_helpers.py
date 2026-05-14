import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from pyAgxArm import AgxArmFactory, ArmModel, PiperFW, create_agx_arm_config


FIRMWARE = {
    "default": PiperFW.DEFAULT,
    "v183": PiperFW.V183,
    "v188": PiperFW.V188,
}


def create_piper_robot(
    *,
    firmware: str = "default",
    interface: str = "socketcan",
    channel: str = "can0",
):
    cfg = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version=FIRMWARE[firmware],
        interface=interface,
        channel=channel,
    )
    return AgxArmFactory.create_arm(cfg)


def enable_robot(robot, timeout_s: float = 5.0, poll_s: float = 0.05) -> bool:
    start_t = time.monotonic()
    while time.monotonic() - start_t < timeout_s:
        if robot.enable():
            return True
        time.sleep(poll_s)
    return False


def disable_robot(robot, attempts: int = 5, poll_s: float = 0.1) -> bool:
    for _ in range(attempts):
        if robot.disable():
            return True
        time.sleep(poll_s)
    return False


def _status_value(value: object) -> str:
    return "" if value is None else str(value)


def collect_robot_state(robot) -> Dict[str, object]:
    state: Dict[str, object] = {
        "is_ok": str(robot.is_ok()),
        "receive_fps": f"{robot.get_fps():.3f}",
        "status_timestamp": "",
        "status_hz": "",
        "ctrl_mode": "",
        "arm_status": "",
        "mode_feedback": "",
        "teach_status": "",
        "motion_status": "",
        "trajectory_num": "",
        "err_code": "",
        "x": "",
        "y": "",
        "z": "",
        "roll": "",
        "pitch": "",
        "yaw": "",
    }
    for i in range(1, 7):
        state[f"joint_{i}"] = ""

    arm_status = robot.get_arm_status()
    if arm_status is not None:
        msg = arm_status.msg
        state.update(
            {
                "status_timestamp": f"{arm_status.timestamp:.6f}",
                "status_hz": f"{arm_status.hz:.3f}",
                "ctrl_mode": _status_value(msg.ctrl_mode),
                "arm_status": _status_value(msg.arm_status),
                "mode_feedback": _status_value(msg.mode_feedback),
                "teach_status": _status_value(msg.teach_status),
                "motion_status": _status_value(msg.motion_status),
                "trajectory_num": msg.trajectory_num,
                "err_code": msg.err_code,
            }
        )

    joints = robot.get_joint_angles()
    if joints is not None:
        for i, value in enumerate(joints.msg, start=1):
            state[f"joint_{i}"] = f"{value:.9f}"

    flange = robot.get_flange_pose()
    if flange is not None:
        for field, value in zip(["x", "y", "z", "roll", "pitch", "yaw"], flange.msg):
            state[field] = f"{value:.9f}"

    return state


def timestamped_recording_path(
    *,
    output: Optional[str] = None,
    directory: str = "scripts/recordings",
    prefix: str = "dagger",
) -> Path:
    if output:
        return Path(output).expanduser()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(directory).expanduser() / f"{prefix}_{stamp}.csv"
