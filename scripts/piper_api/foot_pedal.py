import glob
import os
import select
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


EV_KEY = 0x01
KEY_PRESS = 1
INPUT_EVENT_STRUCT = struct.Struct("llHHI")
PEDAL_HINTS = ("pedal", "foot")


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
        lines.append(f"{device.path}: {label}")
        for link in device.links:
            lines.append(f"  {link}")
    return "\n".join(lines)


def find_foot_pedal_device() -> str:
    candidates = []
    for device in list_input_devices():
        haystack = " ".join([device.path, device.name, *device.links]).lower()
        if any(hint in haystack for hint in PEDAL_HINTS):
            candidates.append(device)

    if len(candidates) == 1:
        return candidates[0].path

    if not candidates:
        raise RuntimeError(
            "Could not auto-detect a foot pedal input device. "
            "Run with --list-input-devices, then pass --foot-pedal-device."
        )

    formatted = format_input_devices(candidates)
    raise RuntimeError(
        "Multiple possible foot pedal devices were found. "
        "Pass --foot-pedal-device with one of these paths:\n"
        f"{formatted}"
    )


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

    @property
    def active_device(self) -> Optional[str]:
        return self.device

    def start(self) -> None:
        if self._fd is not None:
            return
        if self.device is None:
            self.device = find_foot_pedal_device()

        self._fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        self._start_t = time.monotonic()
        self._drain_raw_events()
        self._buffer = b""

    def stop(self) -> None:
        if self._fd is None:
            return
        os.close(self._fd)
        self._fd = None
        self._buffer = b""

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

    def _read_events(self):
        self._drain_raw_events()
        events = []
        event_size = INPUT_EVENT_STRUCT.size
        while len(self._buffer) >= event_size:
            chunk = self._buffer[:event_size]
            self._buffer = self._buffer[event_size:]
            _, _, event_type, code, value = INPUT_EVENT_STRUCT.unpack(chunk)
            events.append((event_type, code, value))
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
