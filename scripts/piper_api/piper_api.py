import math
import time
from dataclasses import dataclass
from typing import Sequence

from .robot_helpers import collect_robot_state, create_piper_robot, disable_robot, enable_robot


JOINT_MODE_ALIASES = {
    "smooth": "smooth",
    "move-j": "smooth",
    "move_j": "smooth",
    "j": "smooth",
    "exact": "exact",
    "move-js": "exact",
    "move_js": "exact",
    "js": "exact",
    "cpv": "cpv",
}

INSTALLATION_POS_ALIASES = {
    None: None,
    "unchanged": None,
    "none": None,
    "horizontal": "HORIZONTAL",
    "up": "HORIZONTAL",
    "left": "LEFT",
    "side-left": "LEFT",
    "side_left": "LEFT",
    "right": "RIGHT",
    "side-right": "RIGHT",
    "side_right": "RIGHT",
}


@dataclass(frozen=True)
class PiperJointState:
    joints: list[float]
    timestamp: float
    hz: float


@dataclass(frozen=True)
class PiperEefState:
    pose: list[float]
    timestamp: float
    hz: float


def _as_float_list(values: Sequence[float], *, name: str, length: int) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of {length} numeric values")

    result = [float(value) for value in values]
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must only contain finite numeric values")
    return result


def _normalize_joint_mode(mode: str) -> str:
    normalized = JOINT_MODE_ALIASES.get(mode.strip().lower())
    if normalized is None:
        raise ValueError("Joint mode must be 'smooth', 'exact', or 'cpv'")
    return normalized


def _normalize_installation_pos(installation_pos: str | None) -> str | None:
    if installation_pos is None:
        return None

    normalized = INSTALLATION_POS_ALIASES.get(installation_pos.strip().lower())
    if normalized is None and installation_pos.strip().lower() not in INSTALLATION_POS_ALIASES:
        raise ValueError("installation_pos must be 'horizontal', 'left', 'right', or None")
    return normalized


def move_joint(robot, joints: Sequence[float], *, mode: str = "exact") -> None:
    target_joints = _as_float_list(joints, name="joints", length=6)
    normalized_mode = _normalize_joint_mode(mode)

    if normalized_mode == "smooth":
        robot.move_j(target_joints)
    elif normalized_mode == "exact":
        robot.move_js(target_joints)
    else:
        for joint_index, joint_pos in enumerate(target_joints, start=1):
            robot.move_cpv_pos(joint_index, joint_pos)


def move_eef(robot, pose: Sequence[float]) -> None:
    target_pose = _as_float_list(pose, name="pose", length=6)
    robot.move_p(target_pose)


def create_piper_api(
    *,
    firmware: str = "default",
    interface: str = "socketcan",
    channel: str = "can0",
    installation_pos: str | None = "right",
):
    return PiperAPI(
        firmware=firmware,
        interface=interface,
        channel=channel,
        installation_pos=installation_pos,
    )


class PiperAPI:
    def __init__(
        self,
        *,
        firmware: str = "default",
        interface: str = "socketcan",
        channel: str = "can0",
        installation_pos: str | None = "right",
        robot=None,
    ):
        self.robot = robot or create_piper_robot(
            firmware=firmware,
            interface=interface,
            channel=channel,
        )
        self.installation_pos = installation_pos

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.disconnect()

    def connect(self, *, settle_s: float = 0.5) -> bool:
        self.robot.connect()
        if settle_s > 0:
            time.sleep(settle_s)
        self.set_installation_pos(self.installation_pos)
        return self.is_ok()

    def disconnect(self) -> None:
        self.robot.disconnect()

    def is_ok(self) -> bool:
        return bool(self.robot.is_ok())

    def get_fps(self) -> float:
        return float(self.robot.get_fps())

    def enable(self, *, timeout_s: float = 5.0) -> bool:
        return enable_robot(self.robot, timeout_s=timeout_s)

    def disable(self) -> bool:
        return disable_robot(self.robot)

    def set_speed_percent(self, speed_percent: int) -> None:
        if not 0 <= speed_percent <= 100:
            raise ValueError("speed_percent must be between 0 and 100")
        self.robot.set_speed_percent(speed_percent)

    def set_installation_pos(self, installation_pos: str | None) -> None:
        normalized = _normalize_installation_pos(installation_pos)
        if normalized is None:
            return
        value = getattr(self.robot.OPTIONS.INSTALLATION_POS, normalized)
        self.robot.set_installation_pos(value)

    def get_joint_angles(self) -> list[float] | None:
        state = self.get_joint_state()
        if state is None:
            return None
        return state.joints

    def get_joint_state(self) -> PiperJointState | None:
        joints = self.robot.get_joint_angles()
        if joints is None:
            return None
        return PiperJointState(
            joints=list(joints.msg),
            timestamp=float(joints.timestamp),
            hz=float(joints.hz),
        )

    def get_eef_pose(self) -> list[float] | None:
        state = self.get_eef_state()
        if state is None:
            return None
        return state.pose

    def get_eef_state(self) -> PiperEefState | None:
        flange = self.robot.get_flange_pose()
        if flange is None:
            return None
        return PiperEefState(
            pose=list(flange.msg),
            timestamp=float(flange.timestamp),
            hz=float(flange.hz),
        )

    def get_state(self) -> dict[str, object]:
        return collect_robot_state(self.robot)

    def move_joint(self, joints: Sequence[float], *, mode: str = "exact") -> None:
        move_joint(self.robot, joints, mode=mode)

    def move_eef(self, pose: Sequence[float]) -> None:
        move_eef(self.robot, pose)
