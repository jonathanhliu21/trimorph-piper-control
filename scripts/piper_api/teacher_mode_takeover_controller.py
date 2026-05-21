import time
from dataclasses import dataclass
from typing import List, Optional

from .teacher_mode import get_teacher_mode


@dataclass(frozen=True)
class TeacherModeTakeoverEvent:
    kind: str
    elapsed_s: float
    message: str
    teach_status: object


class TeacherModeTakeoverController:
    def __init__(self, robot, *, takeover_delay_s: float = 1.0):
        if takeover_delay_s < 0:
            raise ValueError("takeover_delay_s must be 0 or greater")

        self.robot = robot
        self.takeover_delay_s = takeover_delay_s
        self.active = False
        self.record_start_t: Optional[float] = None
        self._start_t = 0.0
        self._last_teacher_mode: Optional[bool] = None

    def start(self) -> None:
        self._start_t = time.monotonic()
        self._last_teacher_mode = None
        self.active = False
        self.record_start_t = None

    def stop(self) -> None:
        self.active = False
        self.record_start_t = None
        self._last_teacher_mode = None

    def should_record_takeover(self) -> bool:
        return (
            self.active
            and self.record_start_t is not None
            and time.monotonic() >= self.record_start_t
        )

    def poll(self) -> List[TeacherModeTakeoverEvent]:
        is_teacher_mode, status = get_teacher_mode(self.robot)
        if self._last_teacher_mode is None:
            self._last_teacher_mode = is_teacher_mode
            if is_teacher_mode:
                return [self._start_takeover(status)]
            return []

        if is_teacher_mode == self._last_teacher_mode:
            return []

        self._last_teacher_mode = is_teacher_mode
        if is_teacher_mode:
            return [self._start_takeover(status)]
        return [self._stop_takeover(status)]

    def _start_takeover(self, status) -> TeacherModeTakeoverEvent:
        now = time.monotonic()
        self.active = True
        self.record_start_t = now + self.takeover_delay_s
        return TeacherModeTakeoverEvent(
            kind="takeover_start",
            elapsed_s=now - self._start_t,
            message=f"Teacher mode active (teach_status={status!r}); takeover enabled.",
            teach_status=status,
        )

    def _stop_takeover(self, status) -> TeacherModeTakeoverEvent:
        now = time.monotonic()
        self.active = False
        self.record_start_t = None
        return TeacherModeTakeoverEvent(
            kind="takeover_stop",
            elapsed_s=now - self._start_t,
            message=f"Teacher mode inactive (teach_status={status!r}); takeover ended.",
            teach_status=status,
        )
