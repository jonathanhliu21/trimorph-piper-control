from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path


TRIMORPH_MOTOR_IDS = tuple(range(13))
RAW_EEF_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")


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
    leader_hand_config: Path
    leader_hand_port: str
    leader_measure_rate_hz: float
    follower_hand_config: Path
    follower_hand_port: str
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
        leader_hand_config=Path("config/leader_hand.yaml"),
        leader_hand_port="/dev/serial/by-id/REPLACE_WITH_TRIMORPH_LEADER",
        leader_measure_rate_hz=100.0,
        follower_hand_config=Path("config/follower_hand.yaml"),
        follower_hand_port="/dev/serial/by-id/REPLACE_WITH_TRIMORPH_FOLLOWER",
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
    )


def apply_overrides(
    config: CollectionConfig,
    *,
    dataset_root: str | None = None,
    leader_config: str | None = None,
    leader_port: str | None = None,
    follower_config: str | None = None,
    follower_port: str | None = None,
    foot_pedal_device: str | None = None,
) -> CollectionConfig:
    updates: dict[str, object] = {}
    if dataset_root is not None:
        updates["dataset_root"] = Path(dataset_root).expanduser()
    if leader_config is not None:
        updates["leader_hand_config"] = Path(leader_config).expanduser()
    if leader_port is not None:
        updates["leader_hand_port"] = leader_port
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
        "LEADER_MEASURE_RATE_HZ": config.leader_measure_rate_hz,
        "FOLLOWER_MEASURE_RATE_HZ": config.follower_measure_rate_hz,
        "FOLLOWER_COMMAND_RATE_HZ": config.follower_command_rate_hz,
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
    for label, port in (("leader", config.leader_hand_port), ("follower", config.follower_hand_port)):
        if not port or "REPLACE_WITH" in port:
            raise ValueError(f"Set an explicit TRIMORPH {label} hand serial port")
    if config.leader_hand_port == config.follower_hand_port:
        raise ValueError("Leader and follower hand ports must be different")
    for label, path in (("leader", config.leader_hand_config), ("follower", config.follower_hand_config)):
        if not path.is_file():
            raise FileNotFoundError(f"TRIMORPH {label} hand config not found: {path}")
    if not 0.0 < config.follower_interpolation_alpha <= 1.0:
        raise ValueError("FOLLOWER_INTERPOLATION_ALPHA must be in (0, 1]")
    if config.follower_max_step_rad < 0:
        raise ValueError("FOLLOWER_MAX_STEP_RAD must be non-negative")
    if config.foot_pedal_debounce_seconds < 0:
        raise ValueError("FOOT_PEDAL_DEBOUNCE_SECONDS must be non-negative")
    if require_runtime_modules:
        for module, install_hint in (
            ("depthai", "depthai"),
            ("cv2", "opencv-python"),
            ("midas_hand_api", "an editable install of ../trimorph_hand_api"),
            ("pyAgxArm", "the PiperX vendor Python package"),
        ):
            if not _module_available(module):
                raise RuntimeError(f"Missing {module!r}; install {install_hint}")
