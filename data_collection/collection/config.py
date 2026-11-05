"""Shared collection/replay hardware defaults, retargeting settings, and configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


TRIMORPH_MOTOR_IDS = tuple(range(13))
RAW_EEF_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
QUEST_PACKET_FIELDS = ("seq", "px", "py", "pz", "qx", "qy", "qz", "qw")
DEFAULT_QUEST_HOST = "0.0.0.0"
DEFAULT_QUEST_PORT = 5005
DEFAULT_QUEST_JUMP_THRESHOLD = 0.05
DEFAULT_QUEST_STARTUP_TIMEOUT_S = 5.0
FOLLOWER_PROFILE_VELOCITY_RAD_S = 3.0
FOLLOWER_PROFILE_ACCELERATION_RAD_S2 = 50.0
TASK_CAMERA_WIDTH = 1280
TASK_CAMERA_HEIGHT = 720
TASK_CAMERA_FPS = 30
# One of: "none", "horizontal", "vertical", "both", or "180".
# "180" is the readable alias for flipping both axes.
TASK_CAMERA_FLIP = "both"
# Fixed exposure, sensitivity and white balance (auto-exposure and auto white
# balance are off). Measured from the auto values under the lab lighting on
# 2026-11-01; no frame-to-frame flicker at this exposure.
TASK_CAMERA_EXPOSURE_US = 5829
TASK_CAMERA_ISO = 100
TASK_CAMERA_WHITE_BALANCE_K = 2634
TASK_VIEW_WINDOW_NAME = "trimorph_task_view"
VIDEO_CODEC = "mp4v"


DEFAULT_DATASET_ROOT = Path("data_trimorph")
# Each collection run saves to the next free <name>_<n> folder (n = 0, 1, 2, ...).
DEFAULT_DATASET_NAME = "data_collection_1"
PIPER_JOINT_FIELDS = tuple(f"joint_{joint}" for joint in range(1, 7))


WORKER_INIT_TIMEOUT_S = 60.0
VIDEO_FLUSH_TIMEOUT_S = 5.0
DEFAULT_DEMONSTRATION_DURATION_S = 30.0
DEFAULT_WARMUP_TIME_S = 5.0
DEFAULT_HAND_MEASURE_RATE_HZ = 50.0
DEFAULT_FOLLOWER_MEASURE_RATE_HZ = 100.0
DEFAULT_FORCE_FEEDBACK_RATE_HZ = 50.0
DEFAULT_PIPER_RATE_HZ = 250.0
DEFAULT_PIPER_INTERFACE = "socketcan"
DEFAULT_PIPER_CHANNEL = "can0"
DEFAULT_PIPER_FIRMWARE = "v183"
DEFAULT_PIPER_INSTALLATION_POS = "right"
DEFAULT_PIPER_PAYLOAD = "half"
# Named Piper start configurations (J1..J6 degrees), each the IK solution of a
# dataset's median start flange pose. quest_0: data_trimorph_hand_mocap_quest_0,
# flange about (144, -202, 270) mm. data_trimorph_hand_mocap_quest_4 starts form
# two groups: home_4a, mostly episodes 0-94, flange (230, -170, 349) mm; home_4b,
# episodes 95-220, flange (434, -224, 221) mm. home_4c and home_4d are executor
# starts #1 and #6 of 2026-11-03 (random offsets around home_4b) that worked for
# the policy: flange (418, -232, 198) mm, Euler xyz (165, 33, 97) deg, and
# (453, -229, 203) mm, (157, 27, 88) deg. home_cube1: data_trimorph_hand_mocap_quest_5
# (Vicon poses, one start group), median flange (370, -198, 245) mm, mean Euler xyz
# (-175, 53, 103) deg; starts spread ~30 mm / ~10 deg around it. home_unscrew1:
# data_trimorph_unscrew_cap_0 (Vicon poses, one start group), median flange
# (229, -249, 130) mm, mean Euler xyz (163, -1, 84) deg; starts spread ~20 mm / ~6 deg.
PIPER_START_JOINTS_DEG = {
    "quest_0": (-59.0, 73.0, -49.0, 20.0, 39.0, 19.0),
    "home_4a": (-34.6, 78.6, -72.0, -7.4, 52.9, 57.4),
    "home_4b": (-24.0, 124.9, -106.4, -24.7, 47.8, 72.2),
    "home_4c": (-25.59, 123.74, -98.71, -27.68, 41.75, 69.21),
    "home_4d": (-24.49, 129.76, -111.17, -18.44, 45.25, 71.11),
    "home_cube1": (-19.14, 111.94, -85.65, -58.96, 57.24, 89.30),
    # Previous home_unscrew1:
    # "home_unscrew1": (-50.73, 108.65, -58.83, 22.08, 36.01, 28.63),
    # Updated measured start: flange XYZ m (0.227855, -0.244662, 0.176002),
    # RPY rad (2.742192, 0.222076, 1.525819); joint radians converted to degrees.
    "home_unscrew1": (-49.281017, 99.637991, -56.605022, 17.635011, 28.350983, 26.896988),
}
# Median first processed_encoder_t row (motors 0..12, leader radians) of the same episodes.
HAND_START_POSITIONS_RAD = {
    "quest_0": (0.005, -0.027, 0.663, 0.263, 0.013, -1.217, -0.117, 0.016, -0.225, 0.109, 0.018, 0.013, 0.235),
    "home_4a": (0.009, -0.004, 0.658, 0.368, 0.012, -0.420, 0.103, 0.018, -0.067, 0.025, 0.022, 0.004, 0.094),
    "home_4b": (-0.004, -0.005, 0.628, 0.374, 0.012, -0.292, 0.050, 0.018, -0.017, -0.037, 0.021, -0.028, 0.030),
    "home_cube1": (-0.019, -0.005, 0.730, 0.238, 0.013, -0.213, 0.077, 0.018, -0.159, 0.025, 0.018, -0.070, 0.100),
    "home_unscrew1": (-0.029, -0.137, 0.889, 0.406, 0.015, -1.039, 0.039, 0.013, -0.473, 0.007, -0.152, -0.400, 0.410),
}
# home_4c and home_4d were tested with the home_4b hand start.
HAND_START_POSITIONS_RAD["home_4c"] = HAND_START_POSITIONS_RAD["home_4d"] = HAND_START_POSITIONS_RAD["home_4b"]
DEFAULT_EXECUTOR_START = "home_4c"
DEFAULT_TEACHER_MODE_POLL_HZ = 20.0
DEFAULT_HOLD_DELETE_S = 2.0
DEFAULT_READY_SECONDS = 0.2
DEFAULT_FOOT_PEDAL_DEBOUNCE_S = 0.5
DEFAULT_HAND_CONFIG = Path("config/hand.yaml")
DEFAULT_HAND_PORT = "/dev/serial/by-id/REPLACE_WITH_TRIMORPH_HAND"
DATA_COLLECTION_MODE = "data_collection"
RETARGET_ONLY_MODE = "retarget_only"
DEFAULT_THUMB_MODEL = Path("thumb_retarget_model.json")

PIP_SCALE = 0.6
MCP_SCALE = 0.5

FORCE_FEEDBACK_MOTOR_IDS = (0, 1, *range(4, 13))
# Map each enabled TRIMORPH thumb joint to its MIDAS contact-error source.
FORCE_FEEDBACK_SOURCE_MOTOR_IDS = {
    0: 0,
    1: 1,
}
FORCE_FEEDBACK_MAX_CURRENT_MA = 100.0
FORCE_FEEDBACK_FULL_SCALE_ERROR_RAD = 0.01
TACTILE_FORCE_FEEDBACK_CURRENT_MA = 50.0
TACTILE_FORCE_FEEDBACK_MOTOR_IDS = (
    (4, 5, 6),
    (7, 8, 9),
    (10, 11, 12),
    (0, 1),
)


@dataclass(frozen=True, kw_only=True)
class HandRuntimeConfig:
    """Hand, retargeting, and camera settings used by collection and replay."""

    dataset_root: Path
    hand_config: Path
    hand_port: str
    hand_measure_rate_hz: float
    task_view_window_name: str = TASK_VIEW_WINDOW_NAME
    task_camera_width: int = TASK_CAMERA_WIDTH
    task_camera_height: int = TASK_CAMERA_HEIGHT
    task_camera_fps: int = TASK_CAMERA_FPS
    task_camera_flip: str = TASK_CAMERA_FLIP
    task_camera_exposure_us: int = TASK_CAMERA_EXPOSURE_US
    task_camera_iso: int = TASK_CAMERA_ISO
    task_camera_white_balance_k: int = TASK_CAMERA_WHITE_BALANCE_K
    video_codec: str = VIDEO_CODEC
    active_motors: list[int] = field(
        default_factory=lambda: list(TRIMORPH_MOTOR_IDS)
    )
    follower_profile_velocity_rad_s: float = FOLLOWER_PROFILE_VELOCITY_RAD_S
    follower_profile_acceleration_rad_s2: float = (
        FOLLOWER_PROFILE_ACCELERATION_RAD_S2
    )
    follower_command_scale: float = 1.0
    follower_joint_command_scales: dict[int, float] = field(
        default_factory=lambda: {
            0: 1.0,
            1: 1.0,
            2: 1.0,
            3: 1.0,
            4: PIP_SCALE,
            5: MCP_SCALE,
            6: 1.0,
            7: PIP_SCALE,
            8: MCP_SCALE,
            9: 1.0,
            10: PIP_SCALE,
            11: MCP_SCALE,
            12: 1.0,
        }
    )
    # MIDAS zero offsets; the learned model overrides joints 2 and 3.
    follower_joint_command_offsets: dict[int, float] = field(
        default_factory=lambda: {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0,
            6: 0.0,
            7: 0.0,
            8: 0.0,
            9: 0.0,
            10: 0.0,
            11: 0.0,
            12: 0.262,
        }
    )


@dataclass(frozen=True)
class HandOnlyCollectionConfig(HandRuntimeConfig):
    """Collection worker and episode-control settings."""

    warmup_time: float
    episode_time: float
    use_foot_pedal: bool = False
    hold_delete_s: float = DEFAULT_HOLD_DELETE_S
    ready_seconds: float = DEFAULT_READY_SECONDS
    foot_pedal_device: str | None = None
    foot_pedal_key_code: int | None = None
    foot_pedal_debounce_s: float = DEFAULT_FOOT_PEDAL_DEBOUNCE_S
    mode: str = DATA_COLLECTION_MODE
    show_ui: bool = False
    add_camera: bool = False
    use_piper: bool = False
    use_quest: bool = False
    use_mocap: bool = False
    mocap_topics: tuple[str, ...] = ()
    # Live 3D Quest/Vicon view with signal status; the calibration maps both to Piper axes.
    show_tracking_monitor: bool = False
    monitor_calibration: Path | None = None
    mocap_ik_calibration: Path | None = None
    mocap_ik_topic: str = '/vicon/Yori/Yori'
    mocap_ik_seed_deg: tuple[float, ...] = (0., 45., -45., 0., 30., 0.)
    mocap_ik_start: str = DEFAULT_EXECUTOR_START
    mocap_ik_start_tolerance_deg: float = 5.
    quest_host: str = DEFAULT_QUEST_HOST
    quest_port: int = DEFAULT_QUEST_PORT
    quest_jump_threshold: float = DEFAULT_QUEST_JUMP_THRESHOLD
    quest_startup_timeout_s: float = DEFAULT_QUEST_STARTUP_TIMEOUT_S
    piper_joint_level: bool = False
    piper_rate_hz: float = DEFAULT_PIPER_RATE_HZ
    piper_interface: str = DEFAULT_PIPER_INTERFACE
    piper_channel: str = DEFAULT_PIPER_CHANNEL
    piper_firmware: str = DEFAULT_PIPER_FIRMWARE
    piper_installation_pos: str | None = DEFAULT_PIPER_INSTALLATION_POS
    piper_payload: str | None = DEFAULT_PIPER_PAYLOAD
    # Commanded joint start pose (deg) and the flange pose measured there (m, rad).
    piper_start_joints_deg: tuple[float, ...] | None = None
    piper_start_speed_deg_s: float = 10.0
    piper_start_pose: tuple[float, ...] | None = None
    teacher_mode_poll_hz: float = DEFAULT_TEACHER_MODE_POLL_HZ
    add_tactile: bool = False
    paxini_port: str | None = None
    tactile_debug_log: Path | None = None
    force_feedback_motor_ids: tuple[int, ...] = FORCE_FEEDBACK_MOTOR_IDS
    disabled_force_feedback_motor_ids: tuple[int, ...] = ()
    force_feedback_source_motor_ids: dict[int, int] = field(
        default_factory=lambda: dict(FORCE_FEEDBACK_SOURCE_MOTOR_IDS)
    )
    force_feedback_max_current_ma: float = FORCE_FEEDBACK_MAX_CURRENT_MA
    force_feedback_rate_hz: float = DEFAULT_FORCE_FEEDBACK_RATE_HZ
    force_feedback_full_scale_error_rad: float = (
        FORCE_FEEDBACK_FULL_SCALE_ERROR_RAD
    )
    tactile_force_feedback_current_ma: float = TACTILE_FORCE_FEEDBACK_CURRENT_MA
    tactile_force_feedback_motor_ids: tuple[tuple[int, ...], ...] = (
        TACTILE_FORCE_FEEDBACK_MOTOR_IDS
    )
    follower_hand_config: Path | None = None
    follower_hand_port: str | None = None
    follower_measure_rate_hz: float = DEFAULT_FOLLOWER_MEASURE_RATE_HZ
    follower_command_rate_hz: float = 50.0
    follower_configure_hardware: bool = True
    follower_max_step_rad: float = 0.15
    follower_interpolation_alpha: float = 0.4
    thumb_retarget_model: Path | None = None


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


def validate_camera_settings(config) -> None:
    if (
        config.task_camera_width <= 0
        or config.task_camera_height <= 0
        or config.task_camera_fps <= 0
    ):
        raise ValueError("Task camera dimensions and FPS must be greater than 0")
    if config.task_camera_flip not in {
        "none",
        "horizontal",
        "vertical",
        "both",
        "180",
    }:
        raise ValueError(
            "TASK_CAMERA_FLIP must be one of: none, horizontal, vertical, both, 180"
        )
    if not 0 < config.task_camera_exposure_us <= 1_000_000 / config.task_camera_fps:
        raise ValueError(
            "TASK_CAMERA_EXPOSURE_US must be greater than 0 and fit in one frame"
        )
    if not 100 <= config.task_camera_iso <= 1600:
        raise ValueError("TASK_CAMERA_ISO must be in range 100..1600")
    if not 1000 <= config.task_camera_white_balance_k <= 12000:
        raise ValueError("TASK_CAMERA_WHITE_BALANCE_K must be in range 1000..12000")
    if len(config.video_codec) != 4:
        raise ValueError("VIDEO_CODEC must be a 4-character fourcc")
