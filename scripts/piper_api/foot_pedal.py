import glob
import os
import select
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


EV_KEY = 0x01
KEY_PRESS = 1
KEY_RELEASE = 0
INPUT_EVENT_STRUCT = struct.Struct("llHHI")
PEDAL_HINTS = ("pedal", "foot")
KEYBOARD_HINTS = ("event-kbd", "keyboard")
MOUSE_HINTS = ("event-mouse", "mouse")


@dataclass(frozen=True)
class InputDeviceInfo:
    path: str
    name: str
    links: List[str]


@dataclass(frozen=True)
class FootPedalEvent:
    elapsed_s: float
    code: int
    message: str


def _event_name(path: str) -> str:
    event_name = Path(os.path.realpath(path)).name
    sysfs_name = Path("/sys/class/input") / event_name / "device" / "name"
    try:
        return sysfs_name.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def list_input_devices() -> List[InputDeviceInfo]:
    link_map = {}
    for directory in ("/dev/input/by-id", "/dev/input/by-path"):
        for link in glob.glob(f"{directory}/*"):
            target = os.path.realpath(link)
            if Path(target).name.startswith("event"):
                link_map.setdefault(target, []).append(link)

    paths = set(glob.glob("/dev/input/event*"))
    paths.update(link_map)

    devices = []
    for path in sorted(paths):
        devices.append(
            InputDeviceInfo(
                path=path,
                name=_event_name(path),
                links=sorted(link_map.get(os.path.realpath(path), [])),
            )
        )
    return devices


def format_input_devices(devices: List[InputDeviceInfo]) -> str:
    if not devices:
        return "No /dev/input/event* devices were found."

    lines = []
    for device in devices:
        label = device.name or "(unknown name)"
        readable = "readable" if os.access(_preferred_device_path(device), os.R_OK) else "not readable"
        lines.append(f"{device.path}: {label} [{readable}]")
        for link in device.links:
            lines.append(f"  {link}")
    return "\n".join(lines)


def _device_haystack(device: InputDeviceInfo) -> str:
    return " ".join([device.path, device.name, *device.links]).lower()


def _preferred_device_path(device: InputDeviceInfo) -> str:
    for link in device.links:
        if "/by-id/" in link:
            return link
    return device.path


def _foot_pedal_candidate_score(device: InputDeviceInfo) -> int:
    haystack = _device_haystack(device)
    score = 0
    if "pcsensor" in haystack and "footswitch" in haystack:
        score += 50
    if any(hint in haystack for hint in KEYBOARD_HINTS):
        score += 20
    if any(hint in haystack for hint in MOUSE_HINTS):
        score -= 10
    if os.access(_preferred_device_path(device), os.R_OK):
        score += 5
    if any("/by-id/" in link for link in device.links):
        score += 1
    return score


def find_foot_pedal_candidates() -> List[InputDeviceInfo]:
    candidates = []
    for device in list_input_devices():
        haystack = _device_haystack(device)
        if any(hint in haystack for hint in PEDAL_HINTS):
            candidates.append(device)

    return sorted(
        candidates,
        key=lambda device: (_foot_pedal_candidate_score(device), _preferred_device_path(device)),
        reverse=True,
    )


def find_foot_pedal_device() -> str:
    candidates = find_foot_pedal_candidates()

    if len(candidates) == 1:
        return _preferred_device_path(candidates[0])

    if not candidates:
        raise RuntimeError(
            "Could not auto-detect a foot pedal input device. "
            "Run with --list-input-devices, then pass --foot-pedal-device."
        )

    scored_candidates = [
        (_foot_pedal_candidate_score(device), device) for device in candidates
    ]
    best_score = scored_candidates[0][0]
    best_candidates = [
        device for score, device in scored_candidates if score == best_score
    ]
    if len(best_candidates) == 1 and best_score > 0:
        return _preferred_device_path(best_candidates[0])

    formatted = format_input_devices(candidates)
    raise RuntimeError(
        "Multiple possible foot pedal devices were found. "
        "Pass --foot-pedal-device with one of these paths:\n"
        f"{formatted}"
    )


def _format_open_failures(failures: Sequence[tuple[str, OSError]]) -> str:
    return "\n".join(f"  {path}: {exc}" for path, exc in failures)


class FootPedalTrigger:
    def __init__(
        self,
        device: Optional[str] = None,
        *,
        key_code: Optional[int] = None,
        debounce_s: float = 0.5,
    ):
        if debounce_s < 0:
            raise ValueError("debounce_s must be 0 or greater")
        self.device = device
        self.key_code = key_code
        self.debounce_s = debounce_s

        self._fd: Optional[int] = None
        self._buffer = b""
        self._start_t = 0.0
        self._last_trigger_t = 0.0
        self._pedal_down_t: Optional[float] = None

    @property
    def active_device(self) -> Optional[str]:
        return self.device

    def start(self) -> None:
        if self._fd is not None:
            return

        if self.device is not None:
            paths = [self.device]
            if not Path(self.device).exists():
                paths.extend(
                    _preferred_device_path(device)
                    for device in find_foot_pedal_candidates()
                    if _preferred_device_path(device) != self.device
                )
        else:
            paths = [
                _preferred_device_path(device)
                for device in find_foot_pedal_candidates()
            ]
            if not paths:
                raise RuntimeError(
                    "Could not auto-detect a foot pedal input device. "
                    "Run with --list-input-devices, then pass --foot-pedal-device."
                )

        failures = []
        for path in paths:
            try:
                self._fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError as exc:
                failures.append((path, exc))
                continue
            self.device = path
            break

        if self._fd is None:
            permission_failures = [
                exc for _, exc in failures if isinstance(exc, PermissionError)
            ]
            hint = ""
            if permission_failures:
                hint = (
                    "\nThe foot pedal was detected but is not readable by this user. "
                    "Add your user to the input group, install a udev rule for the "
                    "PCsensor FootSwitch, or pass a readable --foot-pedal-device."
                )
            raise RuntimeError(
                "Could not open any foot pedal input device:\n"
                f"{_format_open_failures(failures)}{hint}"
            )

        self._start_t = time.monotonic()
        self._drain_raw_events()
        self._buffer = b""
        self._pedal_down_t = None

    def stop(self) -> None:
        if self._fd is None:
            return
        os.close(self._fd)
        self._fd = None
        self._buffer = b""
        self._pedal_down_t = None

    def poll(self) -> List[FootPedalEvent]:
        if self._fd is None:
            return []

        events = []
        for event_type, code, value in self._read_events():
            if event_type != EV_KEY:
                continue
            if self.key_code is not None and code != self.key_code:
                continue
            if value != KEY_PRESS:
                continue

            now = time.monotonic()
            if now - self._last_trigger_t < self.debounce_s:
                continue
            self._last_trigger_t = now
            events.append(
                FootPedalEvent(
                    elapsed_s=now - self._start_t,
                    code=code,
                    message=f"Foot pedal pressed on {self.device} (key code {code}).",
                )
            )
        return events

    def current_hold_s(self) -> Optional[float]:
        """Returns how long the pedal has been held in seconds, or None if not pressed."""
        if self._pedal_down_t is None:
            return None
        return time.monotonic() - self._pedal_down_t

    def _read_events(self):
        self._drain_raw_events()
        events = []
        event_size = INPUT_EVENT_STRUCT.size
        while len(self._buffer) >= event_size:
            chunk = self._buffer[:event_size]
            self._buffer = self._buffer[event_size:]
            _, _, event_type, code, value = INPUT_EVENT_STRUCT.unpack(chunk)
            events.append((event_type, code, value))
            if event_type == EV_KEY and (self.key_code is None or code == self.key_code):
                now = time.monotonic()
                if value == KEY_PRESS and self._pedal_down_t is None:
                    self._pedal_down_t = now
                elif value == KEY_RELEASE and self._pedal_down_t is not None:
                    self._pedal_down_t = None
        return events

    def _drain_raw_events(self) -> None:
        if self._fd is None:
            return

        while True:
            readable, _, _ = select.select([self._fd], [], [], 0.0)
            if not readable:
                return
            try:
                data = os.read(self._fd, INPUT_EVENT_STRUCT.size * 64)
            except BlockingIOError:
                return
            if not data:
                return
            self._buffer += data
