from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path

from scripts.midas_api import DEFAULT_CONFIG_PATH


# Operator settings. Adjust these values, then run:
#   python3 data_collection/collect_raw_dataset.py
DATASET_ROOT = Path("data_midas")

READY_SECONDS = 0.2
STOP_EPISODE_ON_TEACHER_MODE_LOST = False

SHOW_TASK_VIEW = True
SHOW_TELEOP_VIEW = True
TASK_VIEW_WINDOW_NAME = "task_view"  # shows the task camera frame (RealSense frame as fallback)
TELEOP_WINDOW_NAME = "midas_hand_teleop"
LOGITECH_WINDOW_NAME = "logitech_finger_cam"  # z-key close-up view of the Logitech teleop camera

# Task camera (OAK-1, via depthai) — the sole source of camera.mp4.
TASK_CAMERA_WIDTH = 1280
TASK_CAMERA_HEIGHT = 720
TASK_CAMERA_FPS = 30
TASK_CAMERA_FLIP = "none"  # one of: "none", "horizontal", "vertical", "both"

# RealSense — used only to supply depth/intrinsics for teleop wrist-pose
# estimation (when TELEOP_SOURCE="realsense"). Never recorded to video, so
# this resolution is independent of TASK_CAMERA_WIDTH/HEIGHT/FPS above; pick
# any profile the device actually supports (check with --list-realsense).
REALSENSE_WIDTH = 640
REALSENSE_HEIGHT = 480
REALSENSE_FPS = 30
REALSENSE_SERIAL: str | None = None
REALSENSE_FLIP = "none"  # one of: "none", "horizontal", "vertical", "both"
REALSENSE_TIMEOUT_MS = 500
REALSENSE_MAX_TIMEOUTS = 30
REALSENSE_WARMUP_FRAMES = 30

VIDEO_CODEC = "mp4v"

RECORD_TELEOP_CAMERA = False
TELEOP_CAMERA: int | str = "/dev/v4l/by-id/usb-046d_0825_5259E8E0-video-index0"  # Logitech UVC Camera (046d:0825)
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
PIPER_FIRMWARE = "v183"  # one of: "default", "v183", "v188"
PIPER_INSTALLATION_POS: str | None = "right"
# Distance (m) from the base_link origin down to the real floor/work surface,
# measured along the physically-vertical axis. For a horizontal (floor)
# mount this is normally ~0 (base_link sits at table height); for a side
# mount (right/left) the arm is raised on a bracket, so this must be
# measured on the actual rig -- do not reuse the floor-mount value.
PIPER_MOUNT_HEIGHT_M = 0.5
PIPER_PAYLOAD: str | None = "half"  # one of: "empty", "half", "full", or None to skip
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

# --------------------------------------------------------------------------
# Camera map — three independent camera "slots", each backed by a physical
# device that can change over time. Match by slot purpose, not by name:
#
#   slot        physical device                            saved to
#   ----------  --------------------------------------    -------------------
#   task        Luxonis OAK-1 camera, via the depthai       camera.mp4
#               SDK (see test_oak1_camera.py). Always        (main training
#               depthai — auto-detected over USB, no          video)
#               /dev/video index to configure. (Formerly
#               a DroidCam via v4l2loopback; see git
#               history if that ever needs to come back.)
#   teleop      Logitech UVC webcam (TELEOP_CAMERA).        teleop_camera.mp4
#               MediaPipe hand-tracking input; primary        (only if
#               source when TELEOP_SOURCE="opencv", or a      RECORD_TELEOP_
#               toggleable close-up "finger cam" (press z)     CAMERA=True)
#               when TELEOP_SOURCE="realsense".
#   aux         Unused by default. A genuinely separate      wrist_camera.mp4
#               camera device, distinct from the task and      (legacy file
#               teleop cameras above — NOT "the camera on       name, kept for
#               the wrist" (that's the task camera). Also       dataset compat)
#               backed by depthai; see aux_camera_worker.py.
#               If you only have one OAK unit, don't enable
#               this at the same time as the task camera —
#               both would try to grab the same device.
#
# RealSense is not a slot of its own: it always runs in the background to
# supply depth/intrinsics for teleop wrist-pose estimation (when
# TELEOP_SOURCE="realsense"), and is never recorded to video.
# --------------------------------------------------------------------------

AUX_CAMERA_ENABLED = False
AUX_CAMERA_WIDTH = 1280
AUX_CAMERA_HEIGHT = 720
AUX_CAMERA_FPS = 30

# cuRobo motion generation (replaces firmware IK when enabled)
CUROBO_ENABLED = False
CUROBO_URDF_PATH = Path(__file__).resolve().parents[2] / "assets" / "models" / "piper_arm.urdf"
CUROBO_XRDF_PATH = Path(__file__).resolve().parents[2] / "assets" / "models" / "piper_arm.xrdf"
CUROBO_INPUT_HZ = 25.0        # EEF pose update rate fed to MotionGen
CUROBO_OUTPUT_HZ = 120.0      # Joint trajectory / MIT command rate
CUROBO_MIT_KP = 15.0          # Position gain, J1-J3 (N·m/rad)
# J4-J6 (wrist) have much lower inertia than J1-J3, and a *higher* kp there
# (previously 30, 2x body) turned out to make things worse, not better:
# JointJitterDiag (piper_worker.py) showed J4-J6 measured-velocity
# peak-to-peak consistently 2-4x higher than J1-J3 throughout an entire
# run, not just transiently. Low-inertia joints have a higher natural
# frequency for a given kp, so stacking extra stiffness on top of already-low
# inertia pushed the wrist into more oscillation, not less. Landed on a
# *lower* wrist kp than body kp instead.

# Dropping wrist kp to 15 (equal to body) and kd to 1.0 barely moved the
# needle: JointJitterDiag still showed J4-J6 measured-velocity peak-to-peak
# running ~2-3x J1-J3 throughout an entire run, essentially unchanged from
# the kp_wrist=30/kd_wrist=1.5 case. Since matching body's gain didn't fix
# it, the wrist's low inertia itself (not "gain higher than body") is the
# main driver -- go lower than body instead of equal to it.
CUROBO_MIT_KP_WRIST = 10.0    # Position gain, J4-J6 (N·m/rad)
# kd=0.5 was tuned against the original kp=5 (ratio 10:1). Raising kp to
# 15/(was 30) without raising kd drops the damping ratio well below that
# baseline, which read as judder during motion. But scaling kd all the way
# up to 1.5/3.0 (matching the kp ratio exactly) instead produced audible
# mid/high-frequency buzzing immediately on MIT-mode entry -- kd amplifies
# velocity-estimate noise directly into torque, so that was noise
# amplification rather than under-damping.
CUROBO_MIT_KD = 0.8           # Velocity gain, J1-J3 (N·m·s/rad)
CUROBO_MIT_KD_WRIST = 0.7     # Velocity gain, J4-J6 (N·m·s/rad)
# Low-pass filter (EMA) applied to measured joint velocity before it feeds
# grav_comp's damping/coulomb/viscous friction-compensation terms -- those
# terms are velocity-dependent, so raw encoder-differentiated velocity noise
# becomes torque noise directly, on top of whatever the wrist's own gains
# contribute. 1.0 = no filtering; lower = heavier smoothing (more lag).
CUROBO_VEL_FILTER_ALPHA = 0.3
CUROBO_GRAV_COMP_DAMPING = 0.018
CUROBO_GRAV_COMP_TORQUE_SCALE = 1.2   # Global gravity torque multiplier (calibrated)
CUROBO_STALE_TIMEOUT_S = 0.5
CUROBO_WORLD_CUBOIDS: list[dict] = []   # Extra obstacles beyond the default table

# Remote cuRobo MPC solve: offload the MPC solve to a second machine's GPU
# (see scripts/curobo_solver_server.py) over a plain TCP socket, instead of
# solving locally. Local solves on this box's RTX 3050 Ti were measured at
# 70-150ms/call (vs. GPU utilization mostly 1-40% -- not GPU-saturation-bound),
# suspected GIL contention from the other CPU-bound worker threads (esp.
# MediaPipe hand tracking) sharing this process. Off by default; the local
# backend (_create_v1_planner/_create_v2_mpc_planner) remains the fallback.
CUROBO_REMOTE_ENABLED = True
CUROBO_REMOTE_HOST = "192.168.10.1"  # LAN IP of the solver box, e.g. "192.168.1.50"
CUROBO_REMOTE_PORT = 5556
CUROBO_REMOTE_TIMEOUT_S = 2.0        # TEMP: widened from 0.15 to diagnose remote solve latency

CANONICAL_MIDAS_MOTOR_IDS = tuple(range(13))
RAW_EEF_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
TACTILE_FINGER_ORDER = ("thumb", "index", "middle", "ring")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


@dataclass(frozen=True)
class CollectionConfig:
    dataset_root: Path
    ready_seconds: float
    stop_episode_on_teacher_mode_lost: bool
    show_task_view: bool
    show_teleop_view: bool
    task_view_window_name: str
    teleop_window_name: str
    logitech_window_name: str
    task_camera_width: int
    task_camera_height: int
    task_camera_fps: int
    task_camera_flip: str
    realsense_width: int
    realsense_height: int
    realsense_fps: int
    realsense_serial: str | None
    realsense_flip: str
    realsense_timeout_ms: int
    realsense_max_timeouts: int
    realsense_warmup_frames: int
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
    piper_mount_height_m: float
    piper_payload: str | None
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
    aux_camera_enabled: bool
    aux_camera_width: int
    aux_camera_height: int
    aux_camera_fps: int
    curobo_enabled: bool
    curobo_urdf_path: Path
    curobo_xrdf_path: Path
    curobo_input_hz: float
    curobo_output_hz: float
    curobo_mit_kp: float
    curobo_mit_kp_wrist: float
    curobo_mit_kd: float
    curobo_mit_kd_wrist: float
    curobo_vel_filter_alpha: float
    curobo_grav_comp_damping: float
    curobo_grav_comp_torque_scale: float
    curobo_stale_timeout_s: float
    curobo_world_cuboids: list
    curobo_remote_enabled: bool
    curobo_remote_host: str
    curobo_remote_port: int
    curobo_remote_timeout_s: float


def build_config() -> CollectionConfig:
    return CollectionConfig(
        dataset_root=Path(DATASET_ROOT).expanduser(),
        ready_seconds=READY_SECONDS,
        stop_episode_on_teacher_mode_lost=STOP_EPISODE_ON_TEACHER_MODE_LOST,
        show_task_view=SHOW_TASK_VIEW,
        show_teleop_view=SHOW_TELEOP_VIEW,
        task_view_window_name=TASK_VIEW_WINDOW_NAME,
        teleop_window_name=TELEOP_WINDOW_NAME,
        logitech_window_name=LOGITECH_WINDOW_NAME,
        task_camera_width=TASK_CAMERA_WIDTH,
        task_camera_height=TASK_CAMERA_HEIGHT,
        task_camera_fps=TASK_CAMERA_FPS,
        task_camera_flip=TASK_CAMERA_FLIP,
        realsense_width=REALSENSE_WIDTH,
        realsense_height=REALSENSE_HEIGHT,
        realsense_fps=REALSENSE_FPS,
        realsense_serial=REALSENSE_SERIAL,
        realsense_flip=REALSENSE_FLIP,
        realsense_timeout_ms=REALSENSE_TIMEOUT_MS,
        realsense_max_timeouts=REALSENSE_MAX_TIMEOUTS,
        realsense_warmup_frames=REALSENSE_WARMUP_FRAMES,
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
        piper_mount_height_m=PIPER_MOUNT_HEIGHT_M,
        piper_payload=PIPER_PAYLOAD,
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
        aux_camera_enabled=AUX_CAMERA_ENABLED,
        aux_camera_width=AUX_CAMERA_WIDTH,
        aux_camera_height=AUX_CAMERA_HEIGHT,
        aux_camera_fps=AUX_CAMERA_FPS,
        curobo_enabled=CUROBO_ENABLED,
        curobo_urdf_path=Path(CUROBO_URDF_PATH).expanduser(),
        curobo_xrdf_path=Path(CUROBO_XRDF_PATH).expanduser(),
        curobo_input_hz=CUROBO_INPUT_HZ,
        curobo_output_hz=CUROBO_OUTPUT_HZ,
        curobo_mit_kp=CUROBO_MIT_KP,
        curobo_mit_kp_wrist=CUROBO_MIT_KP_WRIST,
        curobo_mit_kd=CUROBO_MIT_KD,
        curobo_mit_kd_wrist=CUROBO_MIT_KD_WRIST,
        curobo_vel_filter_alpha=CUROBO_VEL_FILTER_ALPHA,
        curobo_grav_comp_damping=CUROBO_GRAV_COMP_DAMPING,
        curobo_grav_comp_torque_scale=CUROBO_GRAV_COMP_TORQUE_SCALE,
        curobo_stale_timeout_s=CUROBO_STALE_TIMEOUT_S,
        curobo_world_cuboids=list(CUROBO_WORLD_CUBOIDS),
        curobo_remote_enabled=CUROBO_REMOTE_ENABLED,
        curobo_remote_host=CUROBO_REMOTE_HOST,
        curobo_remote_port=CUROBO_REMOTE_PORT,
        curobo_remote_timeout_s=CUROBO_REMOTE_TIMEOUT_S,
    )


def apply_device_overrides(
    config: CollectionConfig,
    *,
    dataset_root: str | None = None,
    realsense_serial: str | None = None,
    teleop_camera: str | None = None,
    foot_pedal_device: str | None = None,
    curobo_enabled: bool | None = None,
    aux_camera_enabled: bool | None = None,
) -> CollectionConfig:
    updates = {}
    if dataset_root is not None:
        updates["dataset_root"] = Path(dataset_root).expanduser()
    if realsense_serial is not None:
        updates["realsense_serial"] = realsense_serial
    if teleop_camera is not None:
        updates["teleop_camera"] = int(teleop_camera) if teleop_camera.isdigit() else teleop_camera
    if foot_pedal_device is not None:
        updates["foot_pedal_device"] = foot_pedal_device
    if curobo_enabled is not None:
        updates["curobo_enabled"] = curobo_enabled
    if aux_camera_enabled is not None:
        updates["aux_camera_enabled"] = aux_camera_enabled
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
    if (
        config.task_camera_width <= 0
        or config.task_camera_height <= 0
        or config.task_camera_fps <= 0
    ):
        raise ValueError("Task camera dimensions and FPS must be greater than 0")
    if config.task_camera_flip not in {"none", "horizontal", "vertical", "both"}:
        raise ValueError("TASK_CAMERA_FLIP must be one of: none, horizontal, vertical, both")
    if (
        config.realsense_width <= 0
        or config.realsense_height <= 0
        or config.realsense_fps <= 0
    ):
        raise ValueError("RealSense dimensions and FPS must be greater than 0")
    if config.realsense_flip not in {"none", "horizontal", "vertical", "both"}:
        raise ValueError("REALSENSE_FLIP must be one of: none, horizontal, vertical, both")
    if len(config.video_codec) != 4:
        raise ValueError("VIDEO_CODEC must be a 4-character fourcc")
    if config.piper_firmware not in {"default", "v183", "v188"}:
        raise ValueError("PIPER_FIRMWARE must be one of: default, v183, v188")
    if not _module_available("depthai"):
        raise ValueError(
            "The task camera (OAK-1) requires the 'depthai' Python package, which "
            "is not installed in this environment. Install it with `pip install "
            "depthai`."
        )
    if config.aux_camera_enabled and not _module_available("depthai"):
        raise ValueError(
            "AUX_CAMERA_ENABLED is true, but the 'depthai' Python package is not "
            "installed in this environment. Install depthai (pip install depthai) "
            "for the OAK-1 auxiliary camera, or run collect_raw_dataset.py "
            "--no-aux-camera to disable it."
        )
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
    if config.curobo_enabled:
        if not _module_available("curobo"):
            raise ValueError(
                "CUROBO_ENABLED is true, but the 'curobo' Python package is not "
                "installed in this environment. Install curobo or run "
                "collect_raw_dataset.py --no-curobo to use firmware IK instead."
            )
        if not (
            _module_available("curobo.geom.sdf.world")
            or _module_available("curobo.model_predictive_control")
        ):
            raise ValueError(
                "CUROBO_ENABLED is true, but the installed curobo package exposes "
                "neither the v1 MotionGen API (curobo.geom.sdf.world) nor the v2 "
                "ModelPredictiveControl API (curobo.model_predictive_control)."
            )
        try:
            import torch
        except Exception as exc:
            raise ValueError(
                "CUROBO_ENABLED is true, but PyTorch could not be imported."
            ) from exc
        if not torch.cuda.is_available():
            raise ValueError(
                "CUROBO_ENABLED is true, but torch.cuda.is_available() is false. "
                "cuRobo requires a working CUDA PyTorch install and NVIDIA driver."
            )
        if not Path(config.curobo_urdf_path).exists():
            raise ValueError(f"CUROBO_URDF_PATH not found: {config.curobo_urdf_path}")
        if config.curobo_output_hz <= 0 or config.curobo_input_hz <= 0:
            raise ValueError("CUROBO_INPUT_HZ and CUROBO_OUTPUT_HZ must be > 0")
        if (
            config.curobo_mit_kp < 0
            or config.curobo_mit_kp_wrist < 0
            or config.curobo_mit_kd < 0
            or config.curobo_mit_kd_wrist < 0
        ):
            raise ValueError(
                "CUROBO_MIT_KP, CUROBO_MIT_KP_WRIST, CUROBO_MIT_KD, and "
                "CUROBO_MIT_KD_WRIST must be >= 0"
            )
        if not (0.0 < config.curobo_vel_filter_alpha <= 1.0):
            raise ValueError("CUROBO_VEL_FILTER_ALPHA must be in (0, 1]")
        if config.curobo_remote_enabled:
            if not config.curobo_remote_host:
                raise ValueError("CUROBO_REMOTE_HOST must be set when CUROBO_REMOTE_ENABLED")
            if config.curobo_remote_port <= 0:
                raise ValueError("CUROBO_REMOTE_PORT must be > 0")
            if config.curobo_remote_timeout_s <= 0:
                raise ValueError("CUROBO_REMOTE_TIMEOUT_S must be > 0")
