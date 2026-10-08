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

PAYLOAD_ALIASES = {
    None: None,
    "unchanged": None,
    "none": None,
    "empty": "EMPTY",
    "no_load": "EMPTY",
    "half": "HALF",
    "full": "FULL",
}


@dataclass(frozen=True)
class PiperJointState:
    joints: list[float]
    timestamp: float
    hz: float


@dataclass(frozen=True)
class PiperMotorState:
    positions: list[float]
    velocities: list[float]
    currents: list[float]
    torques: list[float]
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


def _normalize_payload(payload: str | None) -> str | None:
    if payload is None:
        return None

    normalized = PAYLOAD_ALIASES.get(payload.strip().lower())
    if normalized is None and payload.strip().lower() not in PAYLOAD_ALIASES:
        raise ValueError("payload must be 'empty', 'half', 'full', or None")
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


MIT_TORQUE_LIMITS_BY_FIRMWARE: dict[str, list[float]] = {
    "default": [32.0, 32.0, 32.0, 8.0, 8.0, 8.0],
    "v183": [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
    "v188": [16.0, 16.0, 16.0, 16.0, 16.0, 16.0],
}


def mit_torque_limits_for_firmware(firmware: str) -> list[float]:
    return list(
        MIT_TORQUE_LIMITS_BY_FIRMWARE.get(
            firmware, MIT_TORQUE_LIMITS_BY_FIRMWARE["default"]
        )
    )


def _torque_limit_for_joint(joint_index: int, firmware: str = "default") -> float:
    if joint_index not in range(1, 7):
        raise ValueError("joint_index must be between 1 and 6")
    return mit_torque_limits_for_firmware(firmware)[joint_index - 1]


def create_piper_api(
    *,
    firmware: str = "default",
    interface: str = "socketcan",
    channel: str = "can0",
    installation_pos: str | None = "horizontal",
    payload: str | None = "half",
):
    return PiperAPI(
        firmware=firmware,
        interface=interface,
        channel=channel,
        installation_pos=installation_pos,
        payload=payload,
    )


class PiperAPI:
    def __init__(
        self,
        *,
        firmware: str = "default",
        interface: str = "socketcan",
        channel: str = "can0",
        installation_pos: str | None = "horizontal",
        payload: str | None = "half",
        robot=None,
    ):
        self.firmware = firmware
        self.robot = robot or create_piper_robot(
            firmware=firmware,
            interface=interface,
            channel=channel,
        )
        self.installation_pos = installation_pos
        self.payload = payload

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
        self.set_payload(self.payload)
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

    def set_payload(self, payload: str | None) -> bool:
        normalized = _normalize_payload(payload)
        if normalized is None:
            return True
        value = getattr(self.robot.OPTIONS.PAYLOAD, normalized)
        ok = bool(self.robot.set_payload(value))
        print(f"Piper payload set to {payload!r}: {'ok' if ok else 'no ack'}")
        return ok

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

    def get_motor_state(self) -> PiperMotorState | None:
        motor_states = [self.robot.get_motor_states(i) for i in range(1, 7)]
        if any(state is None for state in motor_states):
            return None

        timestamps = [float(state.timestamp) for state in motor_states]
        hzs = [float(state.hz) for state in motor_states]
        return PiperMotorState(
            positions=[float(state.msg.position) for state in motor_states],
            velocities=[float(state.msg.velocity) for state in motor_states],
            currents=[float(state.msg.current) for state in motor_states],
            torques=[float(state.msg.torque) for state in motor_states],
            timestamp=max(timestamps),
            hz=min(hzs),
        )

    def get_joint_velocities(self) -> list[float] | None:
        state = self.get_motor_state()
        if state is None:
            return None
        return state.velocities

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

    def get_firmware(self, *, timeout_s: float = 5.0) -> dict[str, object] | None:
        firmware = self.robot.get_firmware(timeout=timeout_s)
        if firmware is None:
            return None
        return dict(firmware)

    def get_software_version(self, *, timeout_s: float = 5.0) -> str | None:
        firmware = self.get_firmware(timeout_s=timeout_s)
        if firmware is None:
            return None
        version = firmware.get("software_version")
        return None if version is None else str(version)

    def set_mit_mode(self) -> None:
        self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.MIT)

    def set_position_mode(self) -> None:
        self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.P)

    def move_joint(self, joints: Sequence[float], *, mode: str = "exact") -> None:
        move_joint(self.robot, joints, mode=mode)

    def move_eef(self, pose: Sequence[float]) -> None:
        move_eef(self.robot, pose)

    def move_mit_torque(
        self,
        joint_index: int,
        torque_nm: float,
        *,
        p_des: float = 0.0,
        v_des: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        clamp: bool = True,
    ) -> float:
        if joint_index not in range(1, 7):
            raise ValueError("joint_index must be between 1 and 6")
        if not all(math.isfinite(value) for value in (torque_nm, p_des, v_des, kp, kd)):
            raise ValueError("MIT command values must be finite")

        command_torque = float(torque_nm)
        limit = _torque_limit_for_joint(joint_index, self.firmware)
        if clamp and abs(command_torque) > limit:
            clamped = max(-limit, min(limit, command_torque))
            print(
                f"Warning: joint {joint_index} torque {command_torque:+.3f} N·m "
                f"clamped to {clamped:+.3f} N·m"
            )
            command_torque = clamped

        self.robot.move_mit(
            joint_index=joint_index,
            p_des=float(p_des),
            v_des=float(v_des),
            kp=float(kp),
            kd=float(kd),
            t_ff=command_torque,
        )
        return command_torque

    def move_mit_torques(
        self,
        torques_nm: Sequence[float],
        *,
        p_des: Sequence[float] | None = None,
        v_des: Sequence[float] | None = None,
        kp: float = 0.0,
        kd: float = 0.0,
        clamp: bool = True,
    ) -> list[float]:
        target_torques = _as_float_list(torques_nm, name="torques_nm", length=6)
        target_positions = [0.0] * 6 if p_des is None else _as_float_list(
            p_des, name="p_des", length=6
        )
        target_velocities = [0.0] * 6 if v_des is None else _as_float_list(
            v_des, name="v_des", length=6
        )

        return [
            self.move_mit_torque(
                joint_index,
                torque,
                p_des=target_positions[joint_index - 1],
                v_des=target_velocities[joint_index - 1],
                kp=kp,
                kd=kd,
                clamp=clamp,
            )
            for joint_index, torque in enumerate(target_torques, start=1)
        ]
