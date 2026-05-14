import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import List, Optional

from .robot_helpers import disable_robot


SPACE = " "


@dataclass(frozen=True)
class KeyboardTakeoverEvent:
    kind: str
    elapsed_s: float
    message: str


class KeyboardTakeoverController:
    def __init__(
        self,
        robot,
        *,
        takeover_delay_s: float = 1.0,
        toggle_debounce_s: float = 1.0,
    ):
        if takeover_delay_s < 0:
            raise ValueError("takeover_delay_s must be 0 or greater")
        if toggle_debounce_s < 0:
            raise ValueError("toggle_debounce_s must be 0 or greater")

        self.robot = robot
        self.takeover_delay_s = takeover_delay_s
        self.toggle_debounce_s = toggle_debounce_s

        self.active = False
        self.record_start_t: Optional[float] = None
        self._start_t = 0.0
        self._last_toggle_t = 0.0
        self._stdin_fd: Optional[int] = None
        self._old_terminal_attrs = None

    def start(self) -> None:
        if self._stdin_fd is not None:
            return
        if not sys.stdin.isatty():
            raise RuntimeError("Keyboard takeover requires an interactive terminal")

        self._start_t = time.monotonic()
        self._stdin_fd = sys.stdin.fileno()
        self._old_terminal_attrs = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        termios.tcflush(self._stdin_fd, termios.TCIFLUSH)

    def stop(self) -> None:
        if self._stdin_fd is None:
            return
        if self._old_terminal_attrs is not None:
            termios.tcsetattr(
                self._stdin_fd,
                termios.TCSADRAIN,
                self._old_terminal_attrs,
            )
        self._stdin_fd = None
        self._old_terminal_attrs = None

    def should_record_takeover(self) -> bool:
        return (
            self.active
            and self.record_start_t is not None
            and time.monotonic() >= self.record_start_t
        )

    def poll(self) -> List[KeyboardTakeoverEvent]:
        if self._stdin_fd is None:
            return []

        events: List[KeyboardTakeoverEvent] = []
        for key in self._drain_keys():
            if key != SPACE:
                continue

            now = time.monotonic()
            if now - self._last_toggle_t < self.toggle_debounce_s:
                continue
            self._last_toggle_t = now

            elapsed_s = now - self._start_t
            if self.active:
                self.active = False
                self.record_start_t = None
                events.append(
                    KeyboardTakeoverEvent(
                        kind="takeover_stop",
                        elapsed_s=elapsed_s,
                        message="Space pressed; exiting takeover.",
                    )
                )
            else:
                self.active = True
                self.record_start_t = now + self.takeover_delay_s
                motors_disabled = self._disable_robot()
                if motors_disabled:
                    message = "Space pressed; joint motors disabled for takeover."
                else:
                    message = (
                        "Space pressed; warning: failed to confirm joint motors "
                        "disabled for takeover."
                    )
                events.append(
                    KeyboardTakeoverEvent(
                        kind="takeover_start",
                        elapsed_s=elapsed_s,
                        message=message,
                    )
                )

        return events

    def _disable_robot(self) -> bool:
        if hasattr(self.robot, "robot"):
            disable = getattr(self.robot, "disable", None)
            if callable(disable):
                return bool(disable())
        return disable_robot(self.robot)

    def _drain_keys(self) -> List[str]:
        keys: List[str] = []
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                return keys
            keys.append(sys.stdin.read(1))
