import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import List, Optional

from .foot_pedal import FootPedalTrigger
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
        use_keyboard: bool = True,
        use_foot_pedal: bool = False,
        foot_pedal_device: Optional[str] = None,
        foot_pedal_key_code: Optional[int] = None,
    ):
        if takeover_delay_s < 0:
            raise ValueError("takeover_delay_s must be 0 or greater")
        if toggle_debounce_s < 0:
            raise ValueError("toggle_debounce_s must be 0 or greater")
        if not use_keyboard and not use_foot_pedal:
            raise ValueError("At least one takeover input must be enabled")

        self.robot = robot
        self.takeover_delay_s = takeover_delay_s
        self.toggle_debounce_s = toggle_debounce_s
        self.use_keyboard = use_keyboard
        self.use_foot_pedal = use_foot_pedal
        self.foot_pedal = (
            FootPedalTrigger(
                foot_pedal_device,
                key_code=foot_pedal_key_code,
                debounce_s=toggle_debounce_s,
            )
            if use_foot_pedal
            else None
        )

        self.active = False
        self.record_start_t: Optional[float] = None
        self._start_t = 0.0
        self._last_toggle_t = 0.0
        self._stdin_fd: Optional[int] = None
        self._old_terminal_attrs = None

    def start(self) -> None:
        if self._stdin_fd is not None:
            return
        self._start_t = time.monotonic()

        if self.use_keyboard:
            if not sys.stdin.isatty():
                raise RuntimeError("Keyboard takeover requires an interactive terminal")
            self._stdin_fd = sys.stdin.fileno()
            self._old_terminal_attrs = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            termios.tcflush(self._stdin_fd, termios.TCIFLUSH)

        if self.foot_pedal is not None:
            self.foot_pedal.start()

    def stop(self) -> None:
        if self._stdin_fd is not None and self._old_terminal_attrs is not None:
            termios.tcsetattr(
                self._stdin_fd,
                termios.TCSADRAIN,
                self._old_terminal_attrs,
            )
        self._stdin_fd = None
        self._old_terminal_attrs = None
        if self.foot_pedal is not None:
            self.foot_pedal.stop()

    def should_record_takeover(self) -> bool:
        return (
            self.active
            and self.record_start_t is not None
            and time.monotonic() >= self.record_start_t
        )

    def poll(self) -> List[KeyboardTakeoverEvent]:
        if self._stdin_fd is None and self.foot_pedal is None:
            return []

        events: List[KeyboardTakeoverEvent] = []
        for source_message in self._drain_toggle_messages():
            event = self._toggle(source_message)
            if event is not None:
                events.append(event)

        return events

    def _drain_toggle_messages(self) -> List[str]:
        messages = []
        if self._stdin_fd is not None:
            for key in self._drain_keys():
                if key != SPACE:
                    continue
                messages.append("Space pressed")

        if self.foot_pedal is not None:
            for event in self.foot_pedal.poll():
                messages.append(event.message)
        return messages

    def _toggle(self, source_message: str) -> Optional[KeyboardTakeoverEvent]:
        now = time.monotonic()
        if now - self._last_toggle_t < self.toggle_debounce_s:
            return None
        self._last_toggle_t = now

        elapsed_s = now - self._start_t
        if self.active:
            self.active = False
            self.record_start_t = None
            return KeyboardTakeoverEvent(
                kind="takeover_stop",
                elapsed_s=elapsed_s,
                message=f"{source_message}; exiting takeover.",
            )

        self.active = True
        self.record_start_t = now + self.takeover_delay_s
        motors_disabled = self._disable_robot()
        if motors_disabled:
            message = f"{source_message}; joint motors disabled for takeover."
        else:
            message = (
                f"{source_message}; warning: failed to confirm joint motors "
                "disabled for takeover."
            )
        return KeyboardTakeoverEvent(
            kind="takeover_start",
            elapsed_s=elapsed_s,
            message=message,
        )

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
