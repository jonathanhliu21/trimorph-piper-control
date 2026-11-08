#!/usr/bin/env python3
"""Live Quest versus Vicon positional accuracy using the collection OpenCV UI.

Run from the repository root (with the same environment as collection)::

    python scripts/quest/visualize_quest_accuracy.py --mocap-topic /vicon/Yori/Yori

Rigidly attach the controller and mocap body. Press C, then translate through a
wide area in all three directions during the calibration countdown. A fixed
orthogonal transform (including handedness conversion, without scale fitting)
is fitted once; subsequent samples measure displacement error independently.
This removes the initial offset, so it measures relative tracking accuracy, not
absolute surveyed accuracy. Rotation with separated tracking origins introduces
lever-arm error: use --mocap-offset X Y Z (meters in mocap body axes) to select
the controller's tracking origin. Calibration must include the intended range
of motion; its residual is reported, never subtracted from subsequent errors.

Quest samples are compared to interpolated mocap positions at their host receive
timestamps. Network/sensor latency can therefore appear as error during motion.
No extrapolation across missing frames is allowed. Orange crosses mark suspected
stalls, red segments mark errors strictly above 1 cm. These are diagnostics:
Quest packets contain no occlusion flag. C recalibrates; R clears history;
Esc/Q quits. Drag rotates and right-drag/scroll zooms, as in collection.
"""
from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_collection.collection.config import (
    DEFAULT_QUEST_HOST, DEFAULT_QUEST_PORT, DEFAULT_QUEST_STARTUP_TIMEOUT_S,
)
from data_collection.collection.mocap_worker import run_mocap_worker
from data_collection.collection.quest_worker import QUEST_TRACKER_NAME, run_quest_worker
from data_collection.collection.shared_state import SharedState, TrackerSample
from data_collection.collection.tracking_monitor import (
    AXIS_COLORS, QUEST_COLOR, VICON_COLOR, TrackingMonitor, project,
)
from data_collection.collection.utils import WorkerError, worker_wrapper

ERROR_M = 0.01
RED = (60, 60, 255)
ORANGE = (0, 160, 255)
GRAY = (190, 190, 190)
GREEN = (80, 220, 80)


class PoseHistory(SharedState):
    """Retain worker samples between UI draws, including brief out-and-back errors."""

    def __init__(self):
        super().__init__()
        self.history: dict[str, deque[TrackerSample]] = {}

    def set_tracker_pose(self, name, monotonic_ns, position, quaternion, *, lost=False):
        super().set_tracker_pose(name, monotonic_ns, position, quaternion, lost=lost)
        with self._lock:
            self.history.setdefault(name, deque(maxlen=10000)).append(self._tracker_samples[name])

    def snapshot(self):
        with self._lock:
            return {name: list(samples) for name, samples in self.history.items()}


@dataclass(frozen=True)
class WorkerConfig:
    quest_host: str
    quest_port: int
    quest_startup_timeout_s: float
    mocap_topics: tuple[str, ...]


class NoRecording:
    def current(self):
        return None


def fit_alignment(quest, mocap):
    """Fit mocap = quest @ matrix.T + offset, retaining a possible reflection."""
    quest, mocap = np.asarray(quest, dtype=float), np.asarray(mocap, dtype=float)
    if len(quest) < 20:
        raise ValueError("Need at least 20 synchronized poses; check both streams")
    q, m = quest - quest.mean(axis=0), mocap - mocap.mean(axis=0)
    for points in (q, m):
        singular = np.linalg.svd(points, compute_uv=False) / np.sqrt(len(points))
        if singular[2] < .01:
            raise ValueError("Move at least several cm in THREE directions, then press C again")
    u, _, vt = np.linalg.svd(q.T @ m)
    matrix = (u @ vt).T
    offset = mocap.mean(axis=0) - matrix @ quest.mean(axis=0)
    residual = np.linalg.norm(quest @ matrix.T + offset - mocap, axis=1)
    return matrix, offset, float(np.sqrt(np.mean(residual ** 2)))


def body_position(sample, offset):
    p = np.asarray(sample.position)
    if np.any(offset):
        p = p + Rotation.from_quat(sample.quaternion).apply(offset)
    return p


def synchronized_position(samples, stamp, max_gap_s, offset):
    """Interpolate a valid bracket, rejecting losses and long gaps."""
    index = np.searchsorted([s.monotonic_ns for s in samples], stamp)
    if index < len(samples) and samples[index].monotonic_ns == stamp:
        return None if samples[index].lost else body_position(samples[index], offset)
    if index == 0 or index == len(samples):
        return None
    a, b = samples[index - 1:index + 1]
    if a.lost or b.lost or (b.monotonic_ns - a.monotonic_ns) / 1e9 > max_gap_s:
        return None
    ratio = (stamp - a.monotonic_ns) / (b.monotonic_ns - a.monotonic_ns)
    return body_position(a, offset) * (1 - ratio) + body_position(b, offset) * ratio


def signal_status(sample, now, timeout):
    if sample is None:
        return "NO DATA"
    age = (now - sample.monotonic_ns) / 1e9
    if age > timeout:
        return f"NO SIGNAL / STALLED ({age:.2f}s)"
    if sample.lost:
        return "OBJECT LOST (possible occlusion)"
    return "OK"


def frozen_status(pairs, window_s, epsilon_m, moving_m):
    """Avoid labeling a stationary controller as stalled unless mocap moves."""
    if len(pairs) < 2 or (pairs[-1][0] - pairs[0][0]) / 1e9 < .9 * window_s:
        return ""
    q = np.array([p[1] for p in pairs])
    m = np.array([p[2] for p in pairs])
    q_span = np.linalg.norm(np.ptp(q, axis=0))
    m_span = np.linalg.norm(np.ptp(m, axis=0))
    if q_span < epsilon_m and m_span > moving_m:
        return "QUEST FROZEN while MOCAP MOVES (possible occlusion)"
    if m_span < epsilon_m and q_span > moving_m:
        return "MOCAP FROZEN while QUEST MOVES (possible occlusion)"
    return ""


class AccuracyView(TrackingMonitor):
    """Reuse collection's OpenCV mouse controls and 3D projection."""

    def __init__(self, shared, args):
        super().__init__(shared, show_quest=True, show_vicon=True,
                         window_name="quest_accuracy", size=(1200, 850))
        self.args = args
        self.matrix = self.offset = None
        self.calibration_end = None
        self.calibration_pairs = []
        self.message = "Press C and move in multiple directions to align the streams"
        self.start_ns = time.monotonic_ns()
        self.reset_history()

    def reset_history(self):
        self.trail = deque(maxlen=self.args.history_points)
        self.recent = deque()
        self.last_stamp = 0
        self.count = self.exceedances = 0
        self.sum_squared = self.maximum = 0.
        self.last_error = None
        self.last_exceedance = "none"
        self.last_pair_wall = None
        self.stall = ""
        self.last_event = "none"
        self.previous_event = ""

    def _on_mouse(self, event, x, y, flags, param):
        # Collection double-click normally reanchors each stream independently;
        # preserve the measured alignment here and only reset the camera.
        if event == self.cv2.EVENT_LBUTTONDBLCLK:
            self.yaw, self.pitch, self.scale = math.radians(-35), math.radians(25), 900.
        else:
            super()._on_mouse(event, x, y, flags, param)

    def calibrate(self, now):
        self.matrix = self.offset = None
        self.calibration_pairs = []
        self.calibration_end = now + int(self.args.calibration_seconds * 1e9)
        self.reset_history()
        self.last_stamp = now  # Do not fit old samples retained by the workers.
        self.message = "CALIBRATING: move widely in multiple directions"

    def ingest(self, history, now):
        quest = history.get(QUEST_TRACKER_NAME, [])
        mocap = history.get(self.args.mocap_topic, [])
        for sample in quest:
            stamp = sample.monotonic_ns
            if stamp <= self.last_stamp:
                continue
            # Wait for a future mocap frame to bracket this Quest timestamp.
            if not mocap or stamp > mocap[-1].monotonic_ns:
                if (now - stamp) / 1e9 < self.args.signal_timeout:
                    break
                self.last_stamp = stamp
                continue
            self.last_stamp = stamp
            if sample.lost:
                continue
            m = synchronized_position(mocap, stamp, self.args.max_gap, self.args.mocap_offset)
            q = np.asarray(sample.position)
            if m is None or not np.isfinite(q).all() or not np.isfinite(m).all():
                continue
            self.last_pair_wall = stamp
            if self.recent and (stamp - self.recent[-1][0]) / 1e9 > self.args.max_gap:
                self.recent.clear()
            self.recent.append((stamp, q, m))
            while self.recent and stamp - self.recent[0][0] > self.args.stall_seconds * 1e9:
                self.recent.popleft()
            self.stall = frozen_status(self.recent, self.args.stall_seconds,
                                       self.args.frozen_mm / 1000, self.args.moving_mm / 1000)
            if self.calibration_end is not None:
                if stamp <= self.calibration_end:
                    self.calibration_pairs.append((q, m))
                continue
            if self.matrix is None:
                continue
            aligned = self.matrix @ q + self.offset
            error = float(np.linalg.norm(aligned - m))
            self.last_error = error
            if error > ERROR_M:
                self.last_exceedance = (
                    f"t={(stamp - self.start_ns) / 1e9:.2f}s: {error * 1000:.1f} mm at "
                    f"mocap ({m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f}) m"
                )
            self.count += 1
            self.exceedances += error > ERROR_M
            self.sum_squared += error ** 2
            self.maximum = max(self.maximum, error)
            self.trail.append((stamp, aligned, m, error, bool(self.stall)))
        if self.calibration_end is not None and now >= self.calibration_end:
            self.calibration_end = None
            try:
                q, m = zip(*self.calibration_pairs) if self.calibration_pairs else ([], [])
                self.matrix, self.offset, rms = fit_alignment(q, m)
                self.message = f"Fixed alignment: calibration RMS {rms * 1000:.1f} mm; C to refit"
                if rms > ERROR_M:
                    self.message += " [HIGH residual: check mount/latency]"
                self.reset_history()
                self.last_stamp = now
            except ValueError as exc:
                self.message = str(exc)
            self.calibration_pairs.clear()

    def render(self, now_ns=None):
        cv2 = self.cv2
        now = time.monotonic_ns() if now_ns is None else now_ns
        history = self.shared.snapshot()
        self.ingest(history, now)
        image = np.full((self.height, self.width, 3), 28, dtype=np.uint8)

        def text(value, y, color=GRAY, scale=.55):
            cv2.putText(image, value, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 1, cv2.LINE_AA)

        statuses = []
        for name, label, color in ((QUEST_TRACKER_NAME, "QUEST", QUEST_COLOR),
                                   (self.args.mocap_topic, "MOCAP", VICON_COLOR)):
            samples = history.get(name, [])
            status = signal_status(samples[-1] if samples else None, now, self.args.signal_timeout)
            statuses.append(status)
            text(f"{label}: {status}", 28 + len(statuses) * 27, color if status == "OK" else RED)
        fresh_pair = self.last_pair_wall is not None and (now - self.last_pair_wall) / 1e9 <= self.args.signal_timeout
        error_valid = self.matrix is not None and fresh_pair and all(s == "OK" for s in statuses)
        if self.calibration_end is not None:
            headline = f"CALIBRATING {max(0, (self.calibration_end - now) / 1e9):.1f}s - MOVE IN THREE DIRECTIONS"
        elif not error_valid or self.last_error is None:
            headline = "ERROR UNAVAILABLE: align streams / waiting for synchronized valid poses"
        else:
            headline = f"ERROR {self.last_error * 1000:.1f} mm" + ("  > 1 cm !" if self.last_error > ERROR_M else "  <= 1 cm")
        text(headline, 115, RED if error_valid and self.last_error is not None and self.last_error > ERROR_M else GREEN, .65)
        stall = self.stall if fresh_pair else ""
        text(stall or "Stall test: needs sustained frozen pose AND motion in the other stream", 144, ORANGE)
        if stall and self.trail:
            stamp, q, m, error, _ = self.trail[-1]
            self.trail[-1] = (stamp, q, m, error, True)
        if self.count:
            text(f"N={self.count}   RMS={1000 * math.sqrt(self.sum_squared / self.count):.1f} mm   "
                 f"max={1000 * self.maximum:.1f} mm   >10mm={self.exceedances} ({100 * self.exceedances / self.count:.1f}%)", 173)
        text(self.message, 201)
        event = stall or next((s for s in statuses if s != "OK"), "")
        if event and event != self.previous_event:
            location = ""
            if self.trail:
                p = self.trail[-1][2]
                location = f" at mocap ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m"
                stamp, q, m, error, _ = self.trail[-1]
                self.trail[-1] = (stamp, q, m, error, True)
            self.last_event = f"t={(now - self.start_ns) / 1e9:.2f}s {event}{location}"
            print(self.last_event, flush=True)
        self.previous_event = event
        text(f"Last tracking event: {self.last_event}", 229, ORANGE, .48)
        text(f"Last >1cm: {self.last_exceedance}", 256, RED, .5)
        center = (self.width // 2, 515)
        origin = self.trail[0][2] if self.trail else np.zeros(3)

        def screen(points):
            return project(np.asarray(points) - origin, self.yaw, self.pitch, self.scale, center)

        for v in np.arange(-.5, .501, .1):
            for points in (((v, -.5, 0), (v, .5, 0)), ((-.5, v, 0), (.5, v, 0))):
                a, b = project(points, self.yaw, self.pitch, self.scale, center)
                cv2.line(image, tuple(a), tuple(b), (55, 55, 55), 1, cv2.LINE_AA)
        for i, color in enumerate(AXIS_COLORS):
            a, b = project([np.zeros(3), np.eye(3)[i] * .15], self.yaw, self.pitch, self.scale, center)
            cv2.line(image, tuple(a), tuple(b), color, 2, cv2.LINE_AA)
            cv2.putText(image, "XYZ"[i], tuple(b), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        if self.trail:
            records = list(self.trail)
            qp = screen([r[1] for r in records])
            mp = screen([r[2] for r in records])
            for i, r in enumerate(records):
                if i and (r[0] - records[i - 1][0]) / 1e9 <= self.args.max_gap:
                    cv2.line(image, tuple(qp[i-1]), tuple(qp[i]), QUEST_COLOR, 2, cv2.LINE_AA)
                    cv2.line(image, tuple(mp[i-1]), tuple(mp[i]), VICON_COLOR, 1, cv2.LINE_AA)
            # Draw alerts last so retracing a path cannot erase an earlier error.
            for i, r in enumerate(records):
                if r[3] > ERROR_M:
                    if i and (r[0] - records[i - 1][0]) / 1e9 <= self.args.max_gap:
                        cv2.line(image, tuple(qp[i-1]), tuple(qp[i]), RED, 2, cv2.LINE_AA)
                    cv2.circle(image, tuple(qp[i]), 3, RED, -1, cv2.LINE_AA)
                    cv2.line(image, tuple(qp[i]), tuple(mp[i]), RED, 2, cv2.LINE_AA)
                if r[4]:
                    cv2.drawMarker(image, tuple(mp[i]), ORANGE, cv2.MARKER_TILTED_CROSS, 12, 2)
            cv2.circle(image, tuple(qp[-1]), 6, QUEST_COLOR, -1, cv2.LINE_AA)
            cv2.circle(image, tuple(mp[-1]), 6, VICON_COLOR, -1, cv2.LINE_AA)
            text(f"Last pair t={(records[-1][0] - self.start_ns) / 1e9:.2f}s; "
                 f"mocap XYZ={tuple(round(float(v), 4) for v in records[-1][2])} m", 770)
        text("Quest: orange | mocap: cyan | >1cm: RED path + connectors | stall/loss: orange X", 798)
        text("C: calibrate | R: clear history | Q/Esc: quit | drag: rotate | right-drag/scroll: zoom | grid: 10cm", 827, GRAY, .5)
        return image


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mocap-topic", required=True, help="One Vicon Position topic for the rigidly attached body")
    parser.add_argument("--host", default=DEFAULT_QUEST_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_QUEST_PORT)
    parser.add_argument("--startup-timeout-s", type=float, default=DEFAULT_QUEST_STARTUP_TIMEOUT_S)
    parser.add_argument("--mocap-offset", type=float, nargs=3, default=(0., 0., 0.), metavar=("X", "Y", "Z"))
    parser.add_argument("--calibration-seconds", type=float, default=5.)
    parser.add_argument("--signal-timeout", type=float, default=.5)
    parser.add_argument("--max-gap", type=float, default=.1, help="Maximum mocap interpolation gap in seconds")
    parser.add_argument("--stall-seconds", type=float, default=.5)
    parser.add_argument("--frozen-mm", type=float, default=.2)
    parser.add_argument("--moving-mm", type=float, default=5.)
    parser.add_argument("--history-points", type=int, default=3000, help="Bounded number of paired samples drawn")
    args = parser.parse_args(argv)
    for name in ("startup_timeout_s", "calibration_seconds", "signal_timeout", "max_gap", "stall_seconds", "frozen_mm", "moving_mm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not args.host or not args.mocap_topic or not 1 <= args.port <= 65535:
        parser.error("Require a host, mocap topic and UDP port in 1..65535")
    if args.history_points < 2 or not np.isfinite(args.mocap_offset).all():
        parser.error("History must have at least two points and offsets must be finite")
    if args.moving_mm <= args.frozen_mm:
        parser.error("--moving-mm must exceed --frozen-mm")
    args.mocap_offset = np.asarray(args.mocap_offset)
    return args


def raise_worker_error(errors):
    try:
        error = errors.get_nowait()
    except queue.Empty:
        return
    print(error.traceback_text, file=sys.stderr, end="")
    raise RuntimeError(f"{error.name} worker failed: {error.message}")


def main(argv=None):
    args = parse_args(argv)
    shared = PoseHistory()
    errors: queue.Queue[WorkerError] = queue.Queue()
    config = WorkerConfig(args.host, args.port, args.startup_timeout_s, (args.mocap_topic,))
    view = AccuracyView(shared, args)
    threads = []
    try:
        for name, target in (("quest", run_quest_worker), ("mocap", run_mocap_worker)):
            thread = threading.Thread(target=worker_wrapper, name=f"accuracy-{name}", daemon=True,
                                      args=(name, errors, shared, threading.Event(), target,
                                            config, shared, NoRecording()))
            thread.start()
            threads.append(thread)
        print("Rigidly mount the trackers. Press C to calibrate, then compare motion. Q/Esc quits.")
        while not shared.shutdown.is_set():
            raise_worker_error(errors)
            view.update()
            key = view.cv2.waitKey(30) & 0xff
            if key in (27, ord('q')):
                break
            if key == ord('c'):
                view.calibrate(time.monotonic_ns())
            if key == ord('r'):
                view.reset_history()
                view.last_stamp = time.monotonic_ns()
            if view.cv2.getWindowProperty(view.window_name, view.cv2.WND_PROP_VISIBLE) < 1:
                break
        raise_worker_error(errors)
    except KeyboardInterrupt:
        return 130
    finally:
        shared.shutdown.set()
        for thread in threads:
            thread.join(timeout=5.)
        view.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
