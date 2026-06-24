from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from scripts.midas_api import DEFAULT_CONFIG_PATH


# Operator settings. Adjust these values, then run:
#   python3 data_collection/collect_raw_dataset.py
DATASET_ROOT = Path("data_midas")

READY_SECONDS = 0.2
STOP_EPISODE_ON_TEACHER_MODE_LOST = True

SHOW_REALSENSE_VIEW = True
SHOW_TELEOP_VIEW = True
REALSENSE_WINDOW_NAME = "realsense_task_camera"
TELEOP_WINDOW_NAME = "midas_hand_teleop"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_SERIAL: str | None = None
CAMERA_FLIP = "both"  # one of: "none", "horizontal", "vertical", "both"
CAMERA_TIMEOUT_MS = 500
CAMERA_MAX_TIMEOUTS = 30
CAMERA_WARMUP_FRAMES = 30
VIDEO_CODEC = "mp4v"

RECORD_TELEOP_CAMERA = False
TELEOP_CAMERA: int | str = 6
TELEOP_CAMERA_FPS = 30
TELEOP_HAND_TYPE = "Right"
TELEOP_INPUT_HAND = "auto"
TELEOP_SELFIE = False
TELEOP_MIRROR_INPUT = True
TELEOP_MIRROR_AXIS = "y"
TELEOP_LOCK_INPUT_HAND = False
TELEOP_HAND_LANDMARKER_MODEL: str | None = None
TELEOP_SCALING_FACTOR = 1.15
TELEOP_SOURCE = "realsense"  # one of: "realsense", "opencv"

TELEOP_ENABLE_DEFAULT = False  # toggled at runtime (press 'e' in a display window)

MIDAS_MEASURE_RATE_HZ = 200.0
MIDAS_COMMAND_RATE_HZ = 50.0
MIDAS_HAND_PORT: str | None = None
MIDAS_HAND_BAUDRATE = 4_000_000
MIDAS_HAND_CONFIG = str(DEFAULT_CONFIG_PATH)
MIDAS_CURRENT_LIMIT_MA: int | None = 300
MIDAS_PROFILE_VELOCITY_RAD_S: float | None = 3.0
MIDAS_PROFILE_ACCELERATION_RAD_S2: float | None = 50.0
MIDAS_COMMAND_SCALE = 1.0
MIDAS_MAX_STEP_RAD = 0.15
MIDAS_INTERPOLATION_ALPHA = 0.4
MIDAS_CONFIGURE_HARDWARE = True

PIPER_RATE_HZ = 200.0
PIPER_COMMAND_RATE_HZ = 30.0
PIPER_INTERFACE = "socketcan"
PIPER_CHANNEL = "can0"
PIPER_FIRMWARE = "default"  # one of: "default", "v183", "v188"
PIPER_INSTALLATION_POS: str | None = "right"
TEACHER_MODE_POLL_HZ = 20.0

# Wrist delta teleop: manual camera-to-robot transform and safety parameters.
DELTA_TELEOP_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

# Legacy absolute wrist teleop transform. Kept only for older scripts/configs; the
# live Piper worker now uses DELTA_TELEOP_CONFIG_PATH and relative wrist deltas.
WRIST_CAM_TO_PIPER_EEF_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

ENABLE_TACTILE = True
REQUIRE_TACTILE = True
TACTILE_PORT: str | None = None
TACTILE_RATE_HZ = 60.0
TACTILE_PUBLISH_RATE_HZ = 60.0
TACTILE_FINGERS = ["thumb", "index", "middle", "ring"]

FOOT_PEDAL_DEVICE: str | None = None
FOOT_PEDAL_KEY_CODE: int | None = None
FOOT_PEDAL_DEBOUNCE_SECONDS = 0.5

CANONICAL_MIDAS_MOTOR_IDS = tuple(range(13))
RAW_EEF_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
TACTILE_FINGER_ORDER = ("thumb", "index", "middle", "ring")


@dataclass(frozen=True)
class CollectionConfig:
    dataset_root: Path
    ready_seconds: float
    stop_episode_on_teacher_mode_lost: bool
    show_realsense_view: bool
    show_teleop_view: bool
    realsense_window_name: str
    teleop_window_name: str
    camera_width: int
    camera_height: int
    camera_fps: int
    camera_serial: str | None
    camera_flip: str
    camera_timeout_ms: int
    camera_max_timeouts: int
    camera_warmup_frames: int
    video_codec: str
    record_teleop_camera: bool
    teleop_camera: int | str
    teleop_camera_fps: int
    teleop_hand_type: str
    teleop_input_hand: str
    teleop_selfie: bool
    teleop_mirror_input: bool
    teleop_mirror_axis: str
    teleop_lock_input_hand: bool
    teleop_hand_landmarker_model: str | None
    teleop_scaling_factor: float
    teleop_source: str
    teleop_enable_default: bool
    midas_measure_rate_hz: float
    midas_command_rate_hz: float
    midas_hand_port: str | None
    midas_hand_baudrate: int
    midas_hand_config: str
    midas_current_limit_ma: int | None
    midas_profile_velocity_rad_s: float | None
    midas_profile_acceleration_rad_s2: float | None
    midas_command_scale: float
    midas_max_step_rad: float
    midas_interpolation_alpha: float
    midas_configure_hardware: bool
    piper_rate_hz: float
    piper_command_rate_hz: float
    piper_interface: str
    piper_channel: str
    piper_firmware: str
    piper_installation_pos: str | None
    teacher_mode_poll_hz: float
    delta_teleop_config_path: str
    wrist_cam_to_piper_eef_4x4: list[list[float]]
    enable_tactile: bool
    require_tactile: bool
    tactile_port: str | None
    tactile_rate_hz: float
    tactile_publish_rate_hz: float
    tactile_fingers: list[str]
    foot_pedal_device: str | None
    foot_pedal_key_code: int | None
    foot_pedal_debounce_seconds: float


def build_config() -> CollectionConfig:
    return CollectionConfig(
        dataset_root=Path(DATASET_ROOT).expanduser(),
        ready_seconds=READY_SECONDS,
        stop_episode_on_teacher_mode_lost=STOP_EPISODE_ON_TEACHER_MODE_LOST,
        show_realsense_view=SHOW_REALSENSE_VIEW,
        show_teleop_view=SHOW_TELEOP_VIEW,
        realsense_window_name=REALSENSE_WINDOW_NAME,
        teleop_window_name=TELEOP_WINDOW_NAME,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        camera_fps=CAMERA_FPS,
        camera_serial=CAMERA_SERIAL,
        camera_flip=CAMERA_FLIP,
        camera_timeout_ms=CAMERA_TIMEOUT_MS,
        camera_max_timeouts=CAMERA_MAX_TIMEOUTS,
        camera_warmup_frames=CAMERA_WARMUP_FRAMES,
        video_codec=VIDEO_CODEC,
        record_teleop_camera=RECORD_TELEOP_CAMERA,
        teleop_camera=TELEOP_CAMERA,
        teleop_camera_fps=TELEOP_CAMERA_FPS,
        teleop_hand_type=TELEOP_HAND_TYPE,
        teleop_input_hand=TELEOP_INPUT_HAND,
        teleop_selfie=TELEOP_SELFIE,
        teleop_mirror_input=TELEOP_MIRROR_INPUT,
        teleop_mirror_axis=TELEOP_MIRROR_AXIS,
        teleop_lock_input_hand=TELEOP_LOCK_INPUT_HAND,
        teleop_hand_landmarker_model=TELEOP_HAND_LANDMARKER_MODEL,
        teleop_scaling_factor=TELEOP_SCALING_FACTOR,
        teleop_source=TELEOP_SOURCE,
        teleop_enable_default=TELEOP_ENABLE_DEFAULT,
        midas_measure_rate_hz=MIDAS_MEASURE_RATE_HZ,
        midas_command_rate_hz=MIDAS_COMMAND_RATE_HZ,
        midas_hand_port=MIDAS_HAND_PORT,
        midas_hand_baudrate=MIDAS_HAND_BAUDRATE,
        midas_hand_config=str(Path(MIDAS_HAND_CONFIG).expanduser()),
        midas_current_limit_ma=MIDAS_CURRENT_LIMIT_MA,
        midas_profile_velocity_rad_s=MIDAS_PROFILE_VELOCITY_RAD_S,
        midas_profile_acceleration_rad_s2=MIDAS_PROFILE_ACCELERATION_RAD_S2,
        midas_command_scale=MIDAS_COMMAND_SCALE,
        midas_max_step_rad=MIDAS_MAX_STEP_RAD,
        midas_interpolation_alpha=MIDAS_INTERPOLATION_ALPHA,
        midas_configure_hardware=MIDAS_CONFIGURE_HARDWARE,
        piper_rate_hz=PIPER_RATE_HZ,
        piper_command_rate_hz=PIPER_COMMAND_RATE_HZ,
        piper_interface=PIPER_INTERFACE,
        piper_channel=PIPER_CHANNEL,
        piper_firmware=PIPER_FIRMWARE,
        piper_installation_pos=PIPER_INSTALLATION_POS,
        teacher_mode_poll_hz=TEACHER_MODE_POLL_HZ,
        delta_teleop_config_path=str(Path(DELTA_TELEOP_CONFIG_PATH).expanduser()),
        wrist_cam_to_piper_eef_4x4=[list(row) for row in WRIST_CAM_TO_PIPER_EEF_4X4],
        enable_tactile=ENABLE_TACTILE,
        require_tactile=REQUIRE_TACTILE,
        tactile_port=TACTILE_PORT,
        tactile_rate_hz=TACTILE_RATE_HZ,
        tactile_publish_rate_hz=TACTILE_PUBLISH_RATE_HZ,
        tactile_fingers=list(TACTILE_FINGERS),
        foot_pedal_device=FOOT_PEDAL_DEVICE,
        foot_pedal_key_code=FOOT_PEDAL_KEY_CODE,
        foot_pedal_debounce_seconds=FOOT_PEDAL_DEBOUNCE_SECONDS,
    )


def apply_device_overrides(
    config: CollectionConfig,
    *,
    dataset_root: str | None = None,
    realsense_serial: str | None = None,
    teleop_camera: str | None = None,
    foot_pedal_device: str | None = None,
) -> CollectionConfig:
    updates = {}
    if dataset_root is not None:
        updates["dataset_root"] = Path(dataset_root).expanduser()
    if realsense_serial is not None:
        updates["camera_serial"] = realsense_serial
    if teleop_camera is not None:
        updates["teleop_camera"] = int(teleop_camera) if teleop_camera.isdigit() else teleop_camera
    if foot_pedal_device is not None:
        updates["foot_pedal_device"] = foot_pedal_device
    return replace(config, **updates) if updates else config


def validate_config(config: CollectionConfig) -> None:
    for name, value in {
        "READY_SECONDS": config.ready_seconds,
        "MIDAS_MEASURE_RATE_HZ": config.midas_measure_rate_hz,
        "MIDAS_COMMAND_RATE_HZ": config.midas_command_rate_hz,
        "PIPER_RATE_HZ": config.piper_rate_hz,
        "PIPER_COMMAND_RATE_HZ": config.piper_command_rate_hz,
        "TEACHER_MODE_POLL_HZ": config.teacher_mode_poll_hz,
        "TACTILE_RATE_HZ": config.tactile_rate_hz,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0")
    if config.camera_width <= 0 or config.camera_height <= 0 or config.camera_fps <= 0:
        raise ValueError("Camera dimensions and FPS must be greater than 0")
    if config.camera_flip not in {"none", "horizontal", "vertical", "both"}:
        raise ValueError("CAMERA_FLIP must be one of: none, horizontal, vertical, both")
    if len(config.video_codec) != 4:
        raise ValueError("VIDEO_CODEC must be a 4-character fourcc")
    if config.piper_firmware not in {"default", "v183", "v188"}:
        raise ValueError("PIPER_FIRMWARE must be one of: default, v183, v188")
    if config.foot_pedal_debounce_seconds < 0:
        raise ValueError("FOOT_PEDAL_DEBOUNCE_SECONDS must be 0 or greater")
    if not config.delta_teleop_config_path:
        raise ValueError("DELTA_TELEOP_CONFIG_PATH must not be empty")
    if config.teleop_hand_type not in {"Right", "Left"}:
        raise ValueError("TELEOP_HAND_TYPE must be Right or Left")
    if config.teleop_input_hand not in {"auto", "Right", "Left"}:
        raise ValueError("TELEOP_INPUT_HAND must be auto, Right, or Left")
    if config.teleop_source not in {"realsense", "opencv"}:
        raise ValueError("TELEOP_SOURCE must be one of: realsense, opencv")
    if (
        not isinstance(config.wrist_cam_to_piper_eef_4x4, list)
        or len(config.wrist_cam_to_piper_eef_4x4) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in config.wrist_cam_to_piper_eef_4x4)
    ):
        raise ValueError("WRIST_CAM_TO_PIPER_EEF_4X4 must be a 4x4 nested list")
    if (
        config.midas_profile_velocity_rad_s is not None
        and config.midas_profile_velocity_rad_s < 0
    ):
        raise ValueError("MIDAS_PROFILE_VELOCITY_RAD_S must be 0 or greater, or None")
    if (
        config.midas_profile_acceleration_rad_s2 is not None
        and config.midas_profile_acceleration_rad_s2 < 0
    ):
        raise ValueError(
            "MIDAS_PROFILE_ACCELERATION_RAD_S2 must be 0 or greater, or None"
        )
