from .motion_io import MotionSample, read_motion_csv
from .foot_pedal import (
    FootPedalEvent,
    FootPedalTrigger,
    format_input_devices,
    list_input_devices,
)
from .teacher_mode import get_teacher_mode, is_teacher_mode_status
from .teacher_mode_takeover_controller import (
    TeacherModeTakeoverController,
    TeacherModeTakeoverEvent,
)

__all__ = [
    "MotionSample",
    "read_motion_csv",
    "FootPedalEvent",
    "FootPedalTrigger",
    "format_input_devices",
    "list_input_devices",
    "get_teacher_mode",
    "is_teacher_mode_status",
    "TeacherModeTakeoverController",
    "TeacherModeTakeoverEvent",
]

try:
    from .motion_recorder import DaggerCsvRecorder
    from .keyboard_takeover_controller import (
        KeyboardTakeoverController,
        KeyboardTakeoverEvent,
    )
    from .piper_api import (
        PiperAPI,
        PiperEefState,
        PiperJointState,
        create_piper_api,
        move_eef,
        move_joint,
    )
    from .robot_helpers import (
        collect_robot_state,
        create_piper_robot,
        disable_robot,
        enable_robot,
        timestamped_recording_path,
    )
except ModuleNotFoundError as exc:
    if exc.name != "pyAgxArm":
        raise
else:
    __all__.extend(
        [
            "DaggerCsvRecorder",
            "KeyboardTakeoverController",
            "KeyboardTakeoverEvent",
            "PiperAPI",
            "PiperEefState",
            "PiperJointState",
            "create_piper_api",
            "move_eef",
            "move_joint",
            "collect_robot_state",
            "create_piper_robot",
            "disable_robot",
            "enable_robot",
            "timestamped_recording_path",
        ]
    )
