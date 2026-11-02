"""Live 3D view of the Quest and Vicon poses with a per-stream signal status.

Each stream is drawn relative to its pose when the view started (or was last
re-centered), so both start at the origin. Displacements are shown in the
Piper base axes when a calibration provides the mapping (linear_matrix for the
Quest, vicon_reference for Vicon) and in the tracker's own axes otherwise.
Drag to rotate, right-drag or scroll to zoom, double-click to re-center.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .quest_worker import QUEST_TRACKER_NAME
from .shared_state import SharedState, TrackerSample
from .utils import import_cv2

TRACKING_WINDOW_NAME = "trimorph_tracking"
# Vicon over Wi-Fi arrives in bursts with gaps of up to ~0.3 s, so allow 0.5 s.
SIGNAL_TIMEOUT_S = 0.5
# Without positional tracking the Quest keeps resending its last position, while
# a tracked, hand-held controller moves by more than 0.1 mm every second.
FROZEN_WINDOW_S = 1.0
FROZEN_SPAN_M = 1e-4
TRAIL_S = 10.0
GRID_HALF_M = 0.5
GRID_STEP_M = 0.1
TRIAD_M = 0.06

# BGR colors.
OK_COLOR = (60, 170, 60)
FAIL_COLOR = (40, 40, 210)
QUEST_COLOR = (0, 165, 255)
VICON_COLOR = (255, 200, 0)
AXIS_COLORS = ((60, 60, 230), (60, 200, 60), (230, 120, 40))


@dataclass(frozen=True)
class StreamStatus:
    ok: bool
    text: str


def stream_status(
    now_ns: int,
    sample: TrackerSample | None,
    recent_positions: np.ndarray | None,
    *,
    check_frozen: bool,
    timeout_s: float = SIGNAL_TIMEOUT_S,
) -> StreamStatus:
    """Green only when samples are fresh, the object is tracked, and (for the Quest) moving.

    recent_positions holds the positions seen over the last FROZEN_WINDOW_S, or
    None until that window has filled.
    """

    if sample is None:
        return StreamStatus(False, "NO DATA")
    age_s = (now_ns - sample.monotonic_ns) / 1e9
    if age_s > timeout_s:
        return StreamStatus(False, f"NO SIGNAL {age_s:.1f}s")
    if sample.lost:
        return StreamStatus(False, "OBJECT LOST")
    if (
        check_frozen
        and recent_positions is not None
        and np.ptp(recent_positions, axis=0).max() < FROZEN_SPAN_M
    ):
        return StreamStatus(False, "POSITION FROZEN")
    return StreamStatus(True, "OK")


@dataclass(frozen=True)
class DisplayFrame:
    """Maps native displacements (linear) and rotations (conjugated by rotation) to display axes."""

    linear: np.ndarray
    rotation: np.ndarray
    label: str


def display_frames(calibration: dict | None) -> tuple[DisplayFrame, DisplayFrame]:
    """Quest and Vicon display frames: Piper base axes where the calibration maps them."""

    native = DisplayFrame(np.eye(3), np.eye(3), "own axes")
    quest = vicon = native
    if calibration:
        if "linear_matrix" in calibration:
            quest = DisplayFrame(
                np.asarray(calibration["linear_matrix"], dtype=float),
                np.asarray(calibration["orientation"]["rotation_matrix"], dtype=float),
                "Piper axes",
            )
        reference = calibration.get("vicon_reference") or {}
        if reference:
            world = np.asarray(next(iter(reference.values()))["vicon_to_piper_rotation"], dtype=float)
            vicon = DisplayFrame(world, world, "Piper axes")
    return quest, vicon


def project(points, yaw: float, pitch: float, scale: float, center) -> np.ndarray:
    """Orthographic view of display points (m): yaw about Z, then pitch toward the viewer."""

    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    view = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array(
        [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
    )
    rotated = np.atleast_2d(np.asarray(points, dtype=float)) @ view.T
    return np.column_stack(
        (center[0] + scale * rotated[:, 0], center[1] - scale * rotated[:, 2])
    ).round().astype(np.int32)


class _Stream:
    def __init__(self, name: str, label: str, color, frame: DisplayFrame, check_frozen: bool):
        self.name, self.label, self.color, self.frame = name, label, color, frame
        self.check_frozen = check_frozen
        self.anchor: tuple[np.ndarray, Rotation] | None = None
        self.trail: deque[tuple[int, np.ndarray]] = deque()
        self.recent: deque[tuple[int, np.ndarray]] = deque()
        self.counts: deque[tuple[int, int]] = deque()
        self.last_count = 0
        self.pose: tuple[np.ndarray, np.ndarray] | None = None

    def recenter(self) -> None:
        self.anchor, self.pose = None, None
        self.trail.clear()

    def update(self, now_ns: int, sample: TrackerSample | None) -> StreamStatus:
        if sample is not None:
            position = np.asarray(sample.position)
            self.recent.append((now_ns, position))
            self.counts.append((now_ns, sample.count))
            if sample.count != self.last_count and not sample.lost:
                rotation = Rotation.from_quat(sample.quaternion)
                if self.anchor is None:
                    self.anchor = (position, rotation)
                origin, start = self.anchor
                display = self.frame.linear @ (position - origin)
                relative = (rotation * start.inv()).as_matrix()
                self.pose = (display, self.frame.rotation @ relative @ self.frame.rotation.T)
                self.trail.append((now_ns, display))
            self.last_count = sample.count
        for queue, keep_ns in (
            (self.trail, TRAIL_S * 1e9),
            (self.recent, FROZEN_WINDOW_S * 1e9),
            (self.counts, 1e9),
        ):
            while queue and now_ns - queue[0][0] > keep_ns:
                queue.popleft()
        window_full = bool(self.recent) and now_ns - self.recent[0][0] >= 0.9 * FROZEN_WINDOW_S * 1e9
        recent = np.array([position for _, position in self.recent]) if window_full else None
        status = stream_status(now_ns, sample, recent, check_frozen=self.check_frozen)
        if status.ok and len(self.counts) > 1:
            (t0, c0), (t1, c1) = self.counts[0], self.counts[-1]
            if t1 > t0:
                status = StreamStatus(True, f"OK {1e9 * (c1 - c0) / (t1 - t0):.0f} Hz")
        return status


class TrackingMonitor:
    """OpenCV window drawn from the main thread; call update() from its loop."""

    def __init__(
        self,
        shared: SharedState,
        *,
        show_quest: bool,
        show_vicon: bool,
        recorder=None,
        calibration: dict | None = None,
        window_name: str = TRACKING_WINDOW_NAME,
        size: tuple[int, int] = (760, 620),
        cv2=None,
    ) -> None:
        self.cv2 = cv2 or import_cv2()
        self.shared, self.recorder, self.window_name = shared, recorder, window_name
        self.width, self.height = size
        self.show_quest, self.show_vicon = show_quest, show_vicon
        self.quest_frame, self.vicon_frame = display_frames(calibration)
        self.streams: dict[str, _Stream] = {}
        self.yaw, self.pitch, self.scale = math.radians(-35), math.radians(25), 900.0
        self._drag: tuple[int, int, int, float, float, float] | None = None
        self.cv2.namedWindow(window_name)
        self.cv2.setMouseCallback(window_name, self._on_mouse)

    def _stream(self, name: str) -> _Stream:
        if name not in self.streams:
            if name == QUEST_TRACKER_NAME:
                stream = _Stream(name, "QUEST", QUEST_COLOR, self.quest_frame, check_frozen=True)
            else:
                stream = _Stream(name, f"VICON {name.rstrip('/').split('/')[-1]}", VICON_COLOR,
                                 self.vicon_frame, check_frozen=False)
            self.streams[name] = stream
        return self.streams[name]

    def _on_mouse(self, event, x, y, flags, _param) -> None:
        cv2 = self.cv2
        if event == cv2.EVENT_LBUTTONDBLCLK:
            for stream in self.streams.values():
                stream.recenter()
        elif event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self._drag = (event, x, y, self.yaw, self.pitch, self.scale)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            button, x0, y0, yaw, pitch, scale = self._drag
            if button == cv2.EVENT_LBUTTONDOWN:
                self.yaw = yaw + 0.01 * (x - x0)
                self.pitch = float(np.clip(pitch + 0.01 * (y - y0), -1.55, 1.55))
            else:
                self.scale = float(np.clip(scale * math.exp(-0.01 * (y - y0)), 50.0, 20000.0))
        elif event == getattr(cv2, "EVENT_MOUSEWHEEL", None):
            self.scale = float(np.clip(self.scale * (1.15 if flags > 0 else 1 / 1.15), 50.0, 20000.0))

    def render(self, now_ns: int | None = None) -> np.ndarray:
        cv2 = self.cv2
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        samples = self.shared.latest_tracker_poses()
        names = [QUEST_TRACKER_NAME] if self.show_quest else []
        if self.show_vicon:
            names += sorted(name for name in samples if name != QUEST_TRACKER_NAME)
        statuses = [(self._stream(name), self._stream(name).update(now_ns, samples.get(name)))
                    for name in names]
        if self.show_vicon and not any(name != QUEST_TRACKER_NAME for name in names):
            statuses.append((None, StreamStatus(False, "NO DATA")))

        image = np.full((self.height, self.width, 3), 28, dtype=np.uint8)
        center = (self.width // 2, (self.height + 70) // 2)
        grid = np.arange(-GRID_HALF_M, GRID_HALF_M + 1e-9, GRID_STEP_M)
        for value in grid:
            for line in (((value, -GRID_HALF_M, 0), (value, GRID_HALF_M, 0)),
                         ((-GRID_HALF_M, value, 0), (GRID_HALF_M, value, 0))):
                a, b = project(line, self.yaw, self.pitch, self.scale, center)
                cv2.line(image, tuple(a), tuple(b), (55, 55, 55), 1, cv2.LINE_AA)
        origin = project((0, 0, 0), self.yaw, self.pitch, self.scale, center)[0]
        for axis, (color, name) in enumerate(zip(AXIS_COLORS, "XYZ")):
            tip = project(np.eye(3)[axis] * 0.2, self.yaw, self.pitch, self.scale, center)[0]
            cv2.line(image, tuple(origin), tuple(tip), color, 2, cv2.LINE_AA)
            cv2.putText(image, name, tuple(tip + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        for stream, _status in statuses:
            if stream is None or stream.pose is None:
                continue
            if len(stream.trail) > 1:
                points = project([p for _, p in stream.trail], self.yaw, self.pitch, self.scale, center)
                cv2.polylines(image, [points], False, stream.color, 2, cv2.LINE_AA)
            position, rotation = stream.pose
            tip0 = project(position, self.yaw, self.pitch, self.scale, center)[0]
            for axis, color in enumerate(AXIS_COLORS):
                tip = project(position + TRIAD_M * rotation[:, axis], self.yaw, self.pitch,
                              self.scale, center)[0]
                cv2.line(image, tuple(tip0), tuple(tip), color, 2, cv2.LINE_AA)
            cv2.circle(image, tuple(tip0), 5, stream.color, -1, cv2.LINE_AA)

        self._draw_status_bar(image, statuses, now_ns)
        legend = [f"{stream.label}: {stream.frame.label}" for stream, _ in statuses if stream is not None]
        cv2.putText(image, "   ".join(legend), (12, self.height - 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, "drag: rotate   right-drag/scroll: zoom   double-click: re-center   grid 10 cm",
                    (12, self.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        return image

    def _draw_status_bar(self, image: np.ndarray, statuses, now_ns: int) -> None:
        cv2 = self.cv2
        rec_width = 130
        box_width = (self.width - rec_width - 12 * (len(statuses) + 2)) // max(len(statuses), 1)
        x = 12
        for stream, status in statuses:
            color = OK_COLOR if status.ok else FAIL_COLOR
            cv2.rectangle(image, (x, 10), (x + box_width, 62), color, -1)
            if stream is not None:
                cv2.rectangle(image, (x, 10), (x + 8, 62), stream.color, -1)
            label = stream.label if stream is not None else "VICON"
            cv2.putText(image, label, (x + 16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, status.text, (x + 16, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            x += box_width + 12
        session = self.recorder.current() if self.recorder is not None else None
        rec_x = self.width - rec_width - 12
        if session is not None:
            elapsed = max(0.0, (now_ns - session.start_ns) / 1e9)
            cv2.circle(image, (rec_x + 16, 36), 9, (40, 40, 230), -1, cv2.LINE_AA)
            text, color = f"REC {elapsed:5.1f}s", (255, 255, 255)
        else:
            text, color = "NOT RECORDING", (150, 150, 150)
        cv2.putText(image, text, (rec_x + (32 if session is not None else 0), 41),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 + (session is not None), cv2.LINE_AA)

    def update(self) -> None:
        self.cv2.imshow(self.window_name, self.render())

    def close(self) -> None:
        try:
            self.cv2.destroyWindow(self.window_name)
        except Exception:
            pass


def load_monitor_calibration(path: Path | None) -> dict | None:
    return None if path is None else json.loads(Path(path).read_text())
