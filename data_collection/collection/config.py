from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence


TRIMORPH_MOTOR_IDS = tuple(range(13))
RAW_EEF_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
FOLLOWER_PROFILE_VELOCITY_RAD_S = 3.0
FOLLOWER_PROFILE_ACCELERATION_RAD_S2 = 50.0


def default_follower_joint_command_scales() -> dict[int, float]:
    """Return unity command scales keyed by TRIMORPH motor ID."""

    return {motor_id: 1.0 for motor_id in TRIMORPH_MOTOR_IDS}


def joint_command_scales_for_motor_ids(
    scales: Mapping[int, float], motor_ids: Sequence[int]
) -> tuple[float, ...]:
    """Resolve motor-ID keyed scales in command-vector order.

    Unspecified motor IDs retain unity scaling, which keeps custom or partial
    follower configurations backward compatible.
    """

    return tuple(float(scales.get(int(motor_id), 1.0)) for motor_id in motor_ids)


def joint_command_offsets_for_motor_ids(
    offsets: Mapping[int, float], motor_ids: Sequence[int]
) -> tuple[float, ...]:
    """Resolve MIDAS-zero offsets in motor order, defaulting to zero radians."""

    return tuple(float(offsets.get(int(motor_id), 0.0)) for motor_id in motor_ids)


@dataclass(frozen=True)
class CollectionConfig:
    dataset_root: Path
    ready_seconds: float
    show_task_view: bool
    task_view_window_name: str
    task_camera_width: int
    task_camera_height: int
    task_camera_fps: int
    task_camera_flip: str
    video_codec: str
    hand_config: Path
    hand_port: str
    hand_measure_rate_hz: float
    follower_hand_config: Path | None
    follower_hand_port: str | None
    follower_measure_rate_hz: float
    follower_command_rate_hz: float
    follower_configure_hardware: bool
    follower_command_scale: float
    follower_max_step_rad: float
    follower_interpolation_alpha: float
    piper_rate_hz: float
    piper_interface: str
    piper_channel: str
    piper_firmware: str
    piper_installation_pos: str | None
    piper_payload: str | None
    teacher_mode_poll_hz: float
    foot_pedal_device: str | None
    foot_pedal_key_code: int | None
    foot_pedal_debounce_seconds: float
    active_motors: list[int] = field(
        default_factory=lambda: list(TRIMORPH_MOTOR_IDS)
    )
    follower_joint_command_scales: dict[int, float] = field(
        default_factory=default_follower_joint_command_scales
    )
    force_feedback_motor_ids: tuple[int, ...] = tuple(range(4, 13))
    force_feedback_max_current_ma: float = 100.0
    force_feedback_full_scale_error_rad: float = 0.01
    tactile_force_feedback_current_ma: float = 50.0
    tactile_force_feedback_motor_ids: tuple[tuple[int, ...], ...] = (
        (4, 5, 6),
        (7, 8, 9),
        (10, 11, 12),
        (0, 1, 2, 3),
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def build_config() -> CollectionConfig:
    return CollectionConfig(
        dataset_root=Path("data_trimorph"),
        ready_seconds=0.2,
        show_task_view=True,
        task_view_window_name="trimorph_task_view",
        task_camera_width=1280,
        task_camera_height=720,
        task_camera_fps=30,
        task_camera_flip="none",
        video_codec="mp4v",
        hand_config=Path("config/hand.yaml"),
        hand_port="/dev/serial/by-id/REPLACE_WITH_TRIMORPH_HAND",
        hand_measure_rate_hz=100.0,
        follower_hand_config=None,
        follower_hand_port=None,
        follower_measure_rate_hz=200.0,
        follower_command_rate_hz=50.0,
        follower_configure_hardware=True,
        follower_command_scale=1.0,
        follower_max_step_rad=0.15,
        follower_interpolation_alpha=0.4,
        piper_rate_hz=200.0,
        piper_interface="socketcan",
        piper_channel="can0",
        piper_firmware="v183",
        piper_installation_pos="right",
        piper_payload="half",
        teacher_mode_poll_hz=20.0,
        foot_pedal_device=None,
        foot_pedal_key_code=None,
        foot_pedal_debounce_seconds=0.5,
        follower_joint_command_scales=default_follower_joint_command_scales(),
    )


def apply_overrides(
    config: CollectionConfig,
    *,
    dataset_root: str | None = None,
    hand_config: str | None = None,
    hand_port: str | None = None,
    follower_config: str | None = None,
    follower_port: str | None = None,
    foot_pedal_device: str | None = None,
) -> CollectionConfig:
    updates: dict[str, object] = {}
    if dataset_root is not None:
        updates["dataset_root"] = Path(dataset_root).expanduser()
    if hand_config is not None:
        updates["hand_config"] = Path(hand_config).expanduser()
    if hand_port is not None:
        updates["hand_port"] = hand_port
    if follower_config is not None:
        updates["follower_hand_config"] = Path(follower_config).expanduser()
    if follower_port is not None:
        updates["follower_hand_port"] = follower_port
    if foot_pedal_device is not None:
        updates["foot_pedal_device"] = foot_pedal_device
    return replace(config, **updates) if updates else config


def validate_config(config: CollectionConfig, *, require_runtime_modules: bool = True) -> None:
    for name, value in {
        "READY_SECONDS": config.ready_seconds,
        "HAND_MEASURE_RATE_HZ": config.hand_measure_rate_hz,
        "PIPER_RATE_HZ": config.piper_rate_hz,
        "TEACHER_MODE_POLL_HZ": config.teacher_mode_poll_hz,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0")
    if config.task_camera_width <= 0 or config.task_camera_height <= 0 or config.task_camera_fps <= 0:
        raise ValueError("Task camera dimensions and FPS must be greater than 0")
    if config.task_camera_flip not in {"none", "horizontal", "vertical", "both"}:
        raise ValueError("TASK_CAMERA_FLIP must be one of: none, horizontal, vertical, both")
    if len(config.video_codec) != 4:
        raise ValueError("VIDEO_CODEC must be a 4-character fourcc")
    if config.piper_firmware not in {"default", "v183", "v188"}:
        raise ValueError("PIPER_FIRMWARE must be one of: default, v183, v188")
    if not config.hand_port or "REPLACE_WITH" in config.hand_port:
        raise ValueError("Set an explicit TRIMORPH hand serial port")
    if not config.hand_config.is_file():
        raise FileNotFoundError(f"TRIMORPH hand config not found: {config.hand_config}")
    if config.follower_hand_config is None:
        if config.follower_hand_port is not None:
            raise ValueError("--follower-port requires --follower-config")
    else:
        if not config.follower_hand_port or "REPLACE_WITH" in config.follower_hand_port:
            raise ValueError("Set an explicit follower hand serial port")
        if config.hand_port == config.follower_hand_port:
            raise ValueError("Primary and follower hand ports must be different")
        if not config.follower_hand_config.is_file():
            raise FileNotFoundError(
                f"Follower hand config not found: {config.follower_hand_config}"
            )
        if config.follower_measure_rate_hz <= 0:
            raise ValueError("FOLLOWER_MEASURE_RATE_HZ must be greater than 0")
        if config.follower_command_rate_hz <= 0:
            raise ValueError("FOLLOWER_COMMAND_RATE_HZ must be greater than 0")
        if not math.isfinite(config.follower_command_scale):
            raise ValueError("FOLLOWER_COMMAND_SCALE must be finite")
        if not all(
            math.isfinite(float(scale))
            for scale in config.follower_joint_command_scales.values()
        ):
            raise ValueError("FOLLOWER_JOINT_COMMAND_SCALES must contain finite values")
        if not 0.0 < config.follower_interpolation_alpha <= 1.0:
            raise ValueError("FOLLOWER_INTERPOLATION_ALPHA must be in (0, 1]")
        if config.follower_max_step_rad < 0:
            raise ValueError("FOLLOWER_MAX_STEP_RAD must be non-negative")
    if config.foot_pedal_debounce_seconds < 0:
        raise ValueError("FOOT_PEDAL_DEBOUNCE_SECONDS must be non-negative")
    if require_runtime_modules:
        modules = (
            ("depthai", "depthai"),
            ("cv2", "opencv-python"),
            ("midas_hand_api", "an editable install of ../trimorph_hand_api"),
            ("pyAgxArm", "the PiperX vendor Python package"),
        )
        for module, install_hint in modules:
            if not _module_available(module):
                raise RuntimeError(f"Missing {module!r}; install {install_hint}")
