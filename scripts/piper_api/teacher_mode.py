TEACHER_MODE_ACTIVE_VALUES = {
    "1",
    "true",
    "teach",
    "teacher",
    "teaching",
    "start_recording",
    "start recording",
    "start_recording(0x1)",
    "0x1",
    "enabled",
    "on",
}
TEACHER_MODE_INACTIVE_VALUES = {
    "",
    "0",
    "false",
    "none",
    "normal",
    "disabled",
    "off",
    "stop_recording",
    "stop recording",
    "stop_recording(0x2)",
    "0x2",
    "not teach",
    "not_teacher",
    "not teacher",
    "no_teach",
    "no teach",
}


def is_teacher_mode_status(value) -> bool:
    normalized = str(value).strip().lower()
    if normalized in TEACHER_MODE_ACTIVE_VALUES:
        return True
    if normalized in TEACHER_MODE_INACTIVE_VALUES:
        return False
    if "start_recording" in normalized or "start recording" in normalized:
        return True
    if "stop_recording" in normalized or "stop recording" in normalized:
        return False
    return "teach" in normalized


def get_teacher_mode(robot):
    status = robot.get_state().get("teach_status")
    return is_teacher_mode_status(status), status
