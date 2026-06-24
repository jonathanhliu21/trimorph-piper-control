from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from wrist_pose_estimator import matrix_to_euler_xyz


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class DeltaTeleopConfig:
    r_cam2robot: np.ndarray
    scale_factor: float
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    max_translation_step_m: float
    position_alpha: float
    orientation_alpha: float
    orientation_enabled: bool
    calibration_frames: int


@dataclass(frozen=True)
class DeltaTeleopStatus:
    calibrated: bool
    calibration_count: int
    calibration_frames: int
    frozen: bool
    target_eef: list[float] | None


def _as_array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return arr


def _as_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            return float(text)
        except ValueError:
            return text


def _load_yaml_subset(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line or line.startswith("-"):
            raise ValueError(
                f"{path} uses YAML syntax that requires PyYAML; keep values inline "
                "or install pyyaml."
            )
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid empty key in {path}")
        if not value:
            data[key] = {}
        else:
            data[key] = _parse_scalar(value)
    return data


def load_delta_teleop_config(path: str | Path = DEFAULT_CONFIG_PATH) -> DeltaTeleopConfig:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Delta teleop config not found: {config_path}")

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        raw = _load_yaml_subset(config_path)
    else:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw = {} if loaded is None else dict(loaded)

    required = {
        "r_cam2robot",
        "scale_factor",
        "workspace_min",
        "workspace_max",
        "max_translation_step_m",
        "position_alpha",
        "orientation_alpha",
        "orientation_enabled",
        "calibration_frames",
    }
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Missing delta teleop config keys: {', '.join(missing)}")

    cfg = DeltaTeleopConfig(
        r_cam2robot=_as_array(raw["r_cam2robot"], shape=(3, 3), name="r_cam2robot"),
        scale_factor=_as_float(raw["scale_factor"], name="scale_factor"),
        workspace_min=_as_array(raw["workspace_min"], shape=(3,), name="workspace_min"),
        workspace_max=_as_array(raw["workspace_max"], shape=(3,), name="workspace_max"),
        max_translation_step_m=_as_float(
            raw["max_translation_step_m"],
            name="max_translation_step_m",
        ),
        position_alpha=_as_float(raw["position_alpha"], name="position_alpha"),
        orientation_alpha=_as_float(raw["orientation_alpha"], name="orientation_alpha"),
        orientation_enabled=_as_bool(raw["orientation_enabled"], name="orientation_enabled"),
        calibration_frames=int(raw["calibration_frames"]),
    )
    validate_delta_teleop_config(cfg)
    return cfg


def validate_delta_teleop_config(config: DeltaTeleopConfig) -> None:
    if abs(np.linalg.det(config.r_cam2robot)) < 1e-6:
        raise ValueError("r_cam2robot must be invertible")
    if config.scale_factor <= 0:
        raise ValueError("scale_factor must be greater than 0")
    if np.any(config.workspace_min >= config.workspace_max):
        raise ValueError("workspace_min must be less than workspace_max")
    if config.max_translation_step_m <= 0:
        raise ValueError("max_translation_step_m must be greater than 0")
    if not 0.0 < config.position_alpha <= 1.0:
        raise ValueError("position_alpha must be in (0, 1]")
    if not 0.0 < config.orientation_alpha <= 1.0:
        raise ValueError("orientation_alpha must be in (0, 1]")
    if config.calibration_frames <= 0:
        raise ValueError("calibration_frames must be greater than 0")


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def eef6_to_matrix(eef: Sequence[float]) -> np.ndarray:
    values = [float(v) for v in eef]
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError("EEF pose must contain 6 finite values")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = euler_xyz_to_matrix(values[3], values[4], values[5])
    pose[:3, 3] = np.asarray(values[:3], dtype=np.float64)
    return pose


def matrix_to_eef6(pose: np.ndarray) -> list[float]:
    r = pose[:3, :3]
    t = pose[:3, 3]
    roll, pitch, yaw = matrix_to_euler_xyz(r)
    return [float(t[0]), float(t[1]), float(t[2]), float(roll), float(pitch), float(yaw)]


def _matrix_to_quat(r: np.ndarray) -> np.ndarray:
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.asarray(
            [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = math.sqrt(1.0 + float(r[0, 0] - r[1, 1] - r[2, 2])) * 2.0
            quat = np.asarray(
                [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s],
                dtype=np.float64,
            )
        elif idx == 1:
            s = math.sqrt(1.0 + float(r[1, 1] - r[0, 0] - r[2, 2])) * 2.0
            quat = np.asarray(
                [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s],
                dtype=np.float64,
            )
        else:
            s = math.sqrt(1.0 + float(r[2, 2] - r[0, 0] - r[1, 1])) * 2.0
            quat = np.asarray(
                [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s],
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quat))
    if norm <= 0:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (q / np.linalg.norm(q)).tolist()
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def _average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    q_ref: np.ndarray | None = None
    q_sum = np.zeros(4, dtype=np.float64)
    for r in rotations:
        q = _matrix_to_quat(r)
        if q_ref is None:
            q_ref = q
        elif float(np.dot(q_ref, q)) < 0.0:
            q = -q
        q_sum += q
    norm = float(np.linalg.norm(q_sum))
    if norm <= 0:
        return rotations[-1]
    return _quat_to_matrix(q_sum / norm)


class DeltaTeleopController:
    def __init__(self, config: DeltaTeleopConfig) -> None:
        self.config = config
        self._calibration_wrist: list[np.ndarray] = []
        self._calibration_robot: list[np.ndarray] = []
        self._ref_wrist_pose_cam: np.ndarray | None = None
        self._ref_robot_pose: np.ndarray | None = None
        self._last_target_pose: np.ndarray | None = None
        self._last_target_eef: list[float] | None = None
        self._frozen = False

    @property
    def calibrated(self) -> bool:
        return self._ref_wrist_pose_cam is not None and self._ref_robot_pose is not None

    @property
    def calibration_count(self) -> int:
        return len(self._calibration_wrist)

    def status(self) -> DeltaTeleopStatus:
        return DeltaTeleopStatus(
            calibrated=self.calibrated,
            calibration_count=self.calibration_count,
            calibration_frames=self.config.calibration_frames,
            frozen=self._frozen,
            target_eef=None if self._last_target_eef is None else list(self._last_target_eef),
        )

    def reset_calibration(self) -> None:
        self._calibration_wrist.clear()
        self._calibration_robot.clear()
        self._ref_wrist_pose_cam = None
        self._ref_robot_pose = None
        self._last_target_pose = None
        self._last_target_eef = None
        self._frozen = False

    def set_reference(self, wrist_pose_cam: np.ndarray, robot_eef: Sequence[float]) -> None:
        wrist_pose = self._validate_pose(wrist_pose_cam, name="wrist_pose_cam")
        robot_pose = eef6_to_matrix(robot_eef)
        self._ref_wrist_pose_cam = wrist_pose
        self._ref_robot_pose = robot_pose
        self._last_target_pose = robot_pose.copy()
        self._last_target_eef = matrix_to_eef6(robot_pose)
        self._frozen = False

    def update(
        self,
        wrist_pose_cam: np.ndarray | None,
        robot_eef_for_calibration: Sequence[float] | None = None,
    ) -> list[float] | None:
        wrist_pose = self._coerce_pose(wrist_pose_cam)
        if not self.calibrated:
            if wrist_pose is not None and robot_eef_for_calibration is not None:
                self._add_calibration_sample(wrist_pose, robot_eef_for_calibration)
            return None

        if wrist_pose is None:
            self._frozen = True
            return None if self._last_target_eef is None else list(self._last_target_eef)

        raw_target = self._raw_target_pose(wrist_pose)
        safe_target = self._apply_safety(raw_target)
        self._last_target_pose = safe_target
        self._last_target_eef = matrix_to_eef6(safe_target)
        self._frozen = False
        return list(self._last_target_eef)

    def _add_calibration_sample(
        self,
        wrist_pose: np.ndarray,
        robot_eef: Sequence[float],
    ) -> None:
        robot_pose = eef6_to_matrix(robot_eef)
        self._calibration_wrist.append(wrist_pose.copy())
        self._calibration_robot.append(robot_pose)
        if len(self._calibration_wrist) < self.config.calibration_frames:
            return

        wrist_stack = np.stack(self._calibration_wrist, axis=0)
        robot_stack = np.stack(self._calibration_robot, axis=0)
        ref_wrist = np.eye(4, dtype=np.float64)
        ref_robot = np.eye(4, dtype=np.float64)
        ref_wrist[:3, 3] = wrist_stack[:, :3, 3].mean(axis=0)
        ref_robot[:3, 3] = robot_stack[:, :3, 3].mean(axis=0)
        ref_wrist[:3, :3] = _average_rotation([p[:3, :3] for p in self._calibration_wrist])
        ref_robot[:3, :3] = _average_rotation([p[:3, :3] for p in self._calibration_robot])
        self._ref_wrist_pose_cam = ref_wrist
        self._ref_robot_pose = ref_robot
        self._last_target_pose = ref_robot.copy()
        self._last_target_eef = matrix_to_eef6(ref_robot)

    def _raw_target_pose(self, wrist_pose_cam: np.ndarray) -> np.ndarray:
        assert self._ref_wrist_pose_cam is not None
        assert self._ref_robot_pose is not None

        r_cam2robot = self.config.r_cam2robot
        delta_pos_cam = wrist_pose_cam[:3, 3] - self._ref_wrist_pose_cam[:3, 3]
        delta_pos_robot = r_cam2robot @ delta_pos_cam * self.config.scale_factor

        target = self._ref_robot_pose.copy()
        target[:3, 3] = self._ref_robot_pose[:3, 3] + delta_pos_robot
        if self.config.orientation_enabled:
            delta_rot_cam = self._ref_wrist_pose_cam[:3, :3].T @ wrist_pose_cam[:3, :3]
            delta_rot_robot = r_cam2robot @ delta_rot_cam @ np.linalg.inv(r_cam2robot)
            target[:3, :3] = self._ref_robot_pose[:3, :3] @ delta_rot_robot
        return target

    def _apply_safety(self, pose: np.ndarray) -> np.ndarray:
        target = pose.copy()
        pos = np.clip(target[:3, 3], self.config.workspace_min, self.config.workspace_max)
        if self._last_target_pose is not None:
            delta = pos - self._last_target_pose[:3, 3]
            distance = float(np.linalg.norm(delta))
            if distance > self.config.max_translation_step_m:
                delta = delta / distance * self.config.max_translation_step_m
                pos = self._last_target_pose[:3, 3] + delta
            alpha = self.config.position_alpha
            pos = alpha * pos + (1.0 - alpha) * self._last_target_pose[:3, 3]
        target[:3, 3] = np.clip(pos, self.config.workspace_min, self.config.workspace_max)

        if self._last_target_pose is not None and self.config.orientation_enabled:
            q_prev = _matrix_to_quat(self._last_target_pose[:3, :3])
            q_next = _matrix_to_quat(target[:3, :3])
            q_filtered = _slerp(q_prev, q_next, self.config.orientation_alpha)
            target[:3, :3] = _quat_to_matrix(q_filtered)
        elif not self.config.orientation_enabled and self._ref_robot_pose is not None:
            target[:3, :3] = self._ref_robot_pose[:3, :3]
        return target

    @staticmethod
    def _coerce_pose(pose: np.ndarray | None) -> np.ndarray | None:
        if pose is None:
            return None
        try:
            return DeltaTeleopController._validate_pose(pose, name="pose")
        except ValueError:
            return None

    @staticmethod
    def _validate_pose(pose: np.ndarray, *, name: str) -> np.ndarray:
        arr = np.asarray(pose, dtype=np.float64)
        if arr.shape != (4, 4) or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be a finite 4x4 matrix")
        return arr.copy()
