#!/usr/bin/env python3
"""Live Quest versus Vicon positional accuracy using the collection OpenCV UI.

Run from the repository root (with the same environment as collection)::

    python scripts/quest/visualize_quest_accuracy.py --mocap-topic /vicon/Yori/Yori

Waits indefinitely for both streams by default; the window remains responsive.
Rigidly attach the controller and mocap body. Press C, then spend 20 seconds
translating in all THREE directions and rotating about at least TWO axes.
Calibration jointly estimates a fixed coordinate transform (including handedness,
without scale fitting) and the vector from mocap origin to Quest origin in mocap
body axes. Both remain fixed during measurement. --mocap-offset X Y Z supplies a
known body offset instead, in meters, and disables automatic offset estimation.
Calibration measures relative tracking accuracy, not absolute surveyed accuracy;
its residual is reported, never subtracted from subsequent errors.

Quest samples are compared to interpolated mocap positions at their host receive
timestamps. Network/sensor latency can therefore appear as error during motion.
No extrapolation across missing frames is allowed. Crosses match the affected
tracker's color; red segments mark errors strictly above 1 cm. These are diagnostics:
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
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_collection.collection.config import (
    DEFAULT_QUEST_HOST, DEFAULT_QUEST_PORT,
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
QUEST_MARKER = 1
MOCAP_MARKER = 2


def stall_marker(status):
    if status.startswith("QUEST"):
        return QUEST_MARKER
    if status.startswith("MOCAP"):
        return MOCAP_MARKER
    return 0


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
    quest_startup_timeout_s: float | None
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


def fit_offset_alignment(quest, mocap, rotations):
    """Fit A q + t = m + R d, including unknown body offset d and handedness.

    Centering eliminates t; solve both reflection parities with robust least
    squares. Rotation and translation excitation checks reject ambiguous fits.
    """
    quest, mocap, rotations = (np.asarray(v, dtype=float) for v in (quest, mocap, rotations))
    n = len(quest)
    if n < 20:
        raise ValueError("Need at least 20 synchronized poses; check both streams")
    if (quest.shape != (n, 3) or mocap.shape != (n, 3) or rotations.shape != (n, 3, 3)
            or not all(np.isfinite(v).all() for v in (quest, mocap, rotations))):
        raise ValueError("Calibration requires finite positions and mocap rotations")
    q, m = quest - quest.mean(axis=0), mocap - mocap.mean(axis=0)
    if np.linalg.svd(q, compute_uv=False)[-1] / np.sqrt(n) < .01:
        raise ValueError("Translate several cm in THREE directions, then press C again")
    centered_rotations = rotations - rotations.mean(axis=0)
    rotation_span = np.linalg.svd(centered_rotations.reshape(-1, 3), compute_uv=False) / np.sqrt(n)
    if rotation_span[-1] < .08:
        raise ValueError("Rotate about at least TWO axes (roughly 30+ degrees each), then press C again")
    u, _, vt = np.linalg.svd(q.T @ m)
    initial = (u @ vt).T
    candidates = []
    for parity in (1., -1.):
        reflection = np.diag([1., 1., parity])
        # Closest orthogonal initialization with the requested parity.
        correction = np.diag([1., 1., parity * np.linalg.det(initial)])
        seed = (u @ correction @ vt).T

        def residual(parameters):
            matrix = Rotation.from_rotvec(parameters[:3]).as_matrix() @ reflection
            return (q @ matrix.T - m - centered_rotations @ parameters[3:]).ravel()

        start = np.r_[Rotation.from_matrix(seed @ reflection).as_rotvec(), np.zeros(3)]
        result = least_squares(residual, start, loss="soft_l1", f_scale=.003,
                               x_scale="jac", max_nfev=300)
        if result.success:
            candidates.append((result.cost, result, reflection))
    if not candidates:
        raise ValueError("Offset fit did not converge; repeat with varied translation and rotation")
    _, result, reflection = min(candidates, key=lambda item: item[0])
    singular = np.linalg.svd(result.jac, compute_uv=False) / np.sqrt(n)
    if singular[-1] < .003 or singular[-1] / singular[0] < .005:
        raise ValueError("Calibration motion is ambiguous; vary translation independently of rotation")
    matrix = Rotation.from_rotvec(result.x[:3]).as_matrix() @ reflection
    body_offset = result.x[3:]
    offset = mocap.mean(axis=0) + rotations.mean(axis=0) @ body_offset - matrix @ quest.mean(axis=0)
    errors = quest @ matrix.T + offset - mocap - rotations @ body_offset
    rms = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))
    return matrix, offset, body_offset, rms


def synchronization_issue(samples, stamp, max_gap_s):
    """Explain why fresh individual streams cannot form a valid pair."""
    index = np.searchsorted([s.monotonic_ns for s in samples], stamp)
    if index < len(samples) and samples[index].monotonic_ns == stamp:
        return "mocap pose is lost" if samples[index].lost else ""
    if not samples:
        return "no mocap poses"
    if index == 0:
        return "Quest timestamp precedes mocap history"
    if index == len(samples):
        return "waiting for a newer mocap frame"
    a, b = samples[index - 1:index + 1]
    if a.lost or b.lost:
        return "mocap interpolation bracket includes a lost pose"
    gap = (b.monotonic_ns - a.monotonic_ns) / 1e9
    if gap > max_gap_s:
        return f"mocap gap {gap:.3f}s exceeds --max-gap {max_gap_s:g}s"
    return ""


def synchronized_pose(samples, stamp, max_gap_s):
    """Interpolate position and rotation at the same timestamp, without extrapolation."""
    if synchronization_issue(samples, stamp, max_gap_s):
        return None
    index = np.searchsorted([s.monotonic_ns for s in samples], stamp)
    try:
        if index < len(samples) and samples[index].monotonic_ns == stamp:
            sample = samples[index]
            return np.asarray(sample.position), Rotation.from_quat(sample.quaternion).as_matrix()
        a, b = samples[index - 1:index + 1]
        ratio = (stamp - a.monotonic_ns) / (b.monotonic_ns - a.monotonic_ns)
        ra, rb = Rotation.from_quat(a.quaternion), Rotation.from_quat(b.quaternion)
        rotation = ra * Rotation.from_rotvec((ra.inv() * rb).as_rotvec() * ratio)
        position = np.asarray(a.position) * (1 - ratio) + np.asarray(b.position) * ratio
        return position, rotation.as_matrix()
    except ValueError:  # Invalid orientation cannot locate an offset tracking origin.
        return None


def synchronized_position(samples, stamp, max_gap_s, offset):
    pose = synchronized_pose(samples, stamp, max_gap_s)
    return None if pose is None else pose[0] + pose[1] @ offset


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
        self.body_offset = np.zeros(3) if args.mocap_offset is None else args.mocap_offset.copy()
        self.matrix = self.offset = None
        self.calibration_end = None
        self.calibration_pairs = []
        self.calibration_failure = None
        self.message = "Press C; translate in 3 directions and rotate about 2 axes"
        self.start_ns = time.monotonic_ns()
        self.reset_history()

    def reset_history(self):
        self.trail = deque(maxlen=self.args.history_points)
        self.display_origin = None
        self.recent = deque()
        self.last_stamp = 0
        self.count = self.exceedances = 0
        self.sum_squared = self.maximum = 0.
        self.last_error = None
        self.last_exceedance = "none"
        self.last_pair_wall = None
        self.paired_count = 0
        self.pairing_note = "waiting for samples"
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
        self.calibration_failure = None
        self.calibration_end = now + int(self.args.calibration_seconds * 1e9)
        self.reset_history()
        self.last_stamp = now  # Do not fit old samples retained by the workers.
        self.message = "CALIBRATING: translate in 3 directions AND rotate about 2 axes"

    def ingest(self, history, now):
        quest = history.get(QUEST_TRACKER_NAME, [])
        mocap = history.get(self.args.mocap_topic, [])
        for sample in quest:
            stamp = sample.monotonic_ns
            if stamp <= self.last_stamp:
                continue
            # Wait for a future mocap frame to bracket this Quest timestamp.
            if not mocap or stamp > mocap[-1].monotonic_ns:
                self.pairing_note = "waiting for a newer mocap frame"
                if (now - stamp) / 1e9 < self.args.signal_timeout:
                    break
                self.last_stamp = stamp
                continue
            self.last_stamp = stamp
            if sample.lost:
                continue
            pose = synchronized_pose(mocap, stamp, self.args.max_gap)
            m = None if pose is None else pose[0] + pose[1] @ self.body_offset
            q = np.asarray(sample.position)
            if m is None or not np.isfinite(q).all() or not np.isfinite(m).all():
                self.pairing_note = (synchronization_issue(mocap, stamp, self.args.max_gap)
                                     or "invalid position or mocap quaternion")
                continue
            self.last_pair_wall = stamp
            self.paired_count += 1
            self.pairing_note = "paired successfully"
            if self.recent and (stamp - self.recent[-1][0]) / 1e9 > self.args.max_gap:
                self.recent.clear()
            self.recent.append((stamp, q, m))
            while self.recent and stamp - self.recent[0][0] > self.args.stall_seconds * 1e9:
                self.recent.popleft()
            self.stall = frozen_status(self.recent, self.args.stall_seconds,
                                       self.args.frozen_mm / 1000, self.args.moving_mm / 1000)
            if self.calibration_end is not None:
                if stamp <= self.calibration_end:
                    self.calibration_pairs.append((q, pose[0], pose[1]))
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
            if self.display_origin is None:
                self.display_origin = m.copy()
            self.trail.append((stamp, aligned, m, error, stall_marker(self.stall)))
        if self.calibration_end is not None and now >= self.calibration_end:
            self.calibration_end = None
            try:
                q, m, rotations = zip(*self.calibration_pairs) if self.calibration_pairs else ([], [], [])
                if self.args.mocap_offset is None:
                    matrix, offset, body_offset, rms = fit_offset_alignment(q, m, rotations)
                else:
                    body_offset = self.args.mocap_offset.copy()
                    corrected = np.asarray(m) + np.asarray(rotations) @ body_offset if len(m) else []
                    matrix, offset, rms = fit_alignment(q, corrected)
                self.matrix, self.offset, self.body_offset = matrix, offset, body_offset
                print(f"Mocap body offset XYZ (m): {body_offset.tolist()}; calibration RMS {rms * 1000:.1f} mm", flush=True)
                self.message = f"Fixed alignment: calibration RMS {rms * 1000:.1f} mm; C to refit"
                if rms > ERROR_M:
                    self.message += " [HIGH residual: check mount/latency]"
                self.reset_history()
                self.last_stamp = now
            except ValueError as exc:
                self.calibration_failure = str(exc)
                self.message = f"Calibration failed: {exc}"
                print(self.message, flush=True)
            self.calibration_pairs.clear()

    def unavailable_reason(self, statuses, fresh_pair):
        if self.matrix is None:
            if self.calibration_failure:
                return "CALIBRATION FAILED: see reason below; press C to retry"
            return "CALIBRATION REQUIRED: press C; translate + rotate"
        for label, status in zip(("QUEST", "MOCAP"), statuses):
            if status != "OK":
                return f"ERROR UNAVAILABLE: {label} {status}"
        if not fresh_pair:
            return f"ERROR UNAVAILABLE: {self.pairing_note}"
        if self.last_error is None:
            return "Waiting for first measurement after calibration"
        return ""

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
            headline = f"CALIBRATING {max(0, (self.calibration_end - now) / 1e9):.1f}s - TRANSLATE + ROTATE ABOUT TWO AXES"
        elif not error_valid or self.last_error is None:
            headline = self.unavailable_reason(statuses, fresh_pair)
        else:
            headline = f"ERROR {self.last_error * 1000:.1f} mm" + ("  > 1 cm !" if self.last_error > ERROR_M else "  <= 1 cm")
        text(headline, 115, RED if error_valid and self.last_error is not None and self.last_error > ERROR_M else GREEN if error_valid else ORANGE, .6)
        stall = self.stall if fresh_pair else ""
        text(stall or "Stall test: needs sustained frozen pose AND motion in the other stream", 144, ORANGE)
        markers = stall_marker(stall)
        for flag, status in zip((QUEST_MARKER, MOCAP_MARKER), statuses):
            if status != "OK":
                markers |= flag
        if markers and self.trail:
            stamp, q, m, error, previous_markers = self.trail[-1]
            self.trail[-1] = (stamp, q, m, error, previous_markers | markers)
        if self.count:
            text(f"N={self.count}   RMS={1000 * math.sqrt(self.sum_squared / self.count):.1f} mm   "
                 f"max={1000 * self.maximum:.1f} mm   >10mm={self.exceedances} ({100 * self.exceedances / self.count:.1f}%)", 173)
        else:
            age = "never" if self.last_pair_wall is None else f"{(now - self.last_pair_wall) / 1e9:.2f}s ago"
            text(f"Synchronized poses: {self.paired_count}; last: {age}; {self.pairing_note}", 173, GRAY, .5)
        text(self.message, 201)
        if self.matrix is not None:
            text(f"Mocap body offset XYZ (mm): {tuple(round(float(v * 1000), 1) for v in self.body_offset)}", 282, GRAY, .5)
        event = stall or "; ".join(f"{name}: {status}" for name, status in
                                    zip(("QUEST", "MOCAP"), statuses) if status != "OK")
        if event and event != self.previous_event:
            location = ""
            if self.trail:
                p = self.trail[-1][2]
                location = f" at mocap ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m"
            self.last_event = f"t={(now - self.start_ns) / 1e9:.2f}s {event}{location}"
            print(self.last_event, flush=True)
        self.previous_event = event
        text(f"Last tracking event: {self.last_event}", 229, ORANGE, .48)
        text(f"Last >1cm: {self.last_exceedance}", 256, RED, .5)
        center = (self.width // 2, 515)
        origin = self.display_origin if self.display_origin is not None else np.zeros(3)

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
                if r[4] & QUEST_MARKER:
                    cv2.drawMarker(image, tuple(qp[i]), QUEST_COLOR, cv2.MARKER_TILTED_CROSS, 12, 2)
                if r[4] & MOCAP_MARKER:
                    cv2.drawMarker(image, tuple(mp[i]), VICON_COLOR, cv2.MARKER_TILTED_CROSS, 12, 2)
            cv2.circle(image, tuple(qp[-1]), 6, QUEST_COLOR, -1, cv2.LINE_AA)
            cv2.circle(image, tuple(mp[-1]), 6, VICON_COLOR, -1, cv2.LINE_AA)
            text(f"Last pair t={(records[-1][0] - self.start_ns) / 1e9:.2f}s; "
                 f"mocap XYZ={tuple(round(float(v), 4) for v in records[-1][2])} m", 770)
        text("Quest: orange | mocap: blue | >1cm: RED path + connectors | stall/loss X: tracker color", 798)
        text("C: calibrate | R: clear history | Q/Esc: quit | drag: rotate | right-drag/scroll: zoom | grid: 10cm", 827, GRAY, .5)
        return image


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mocap-topic", required=True, help="One Vicon Position topic for the rigidly attached body")
    parser.add_argument("--host", default=DEFAULT_QUEST_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_QUEST_PORT)
    parser.add_argument("--startup-timeout-s", type=float, default=None,
                        help="Optional Quest startup deadline in seconds; default: wait indefinitely for both streams")
    parser.add_argument("--mocap-offset", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Known mocap-body offset in meters; default: estimate during calibration")
    parser.add_argument("--calibration-seconds", type=float, default=20., help="Motion calibration duration (default: 20 seconds)")
    parser.add_argument("--signal-timeout", type=float, default=.5)
    parser.add_argument("--max-gap", type=float, default=.1, help="Maximum mocap interpolation gap in seconds")
    parser.add_argument("--stall-seconds", type=float, default=.5)
    parser.add_argument("--frozen-mm", type=float, default=.2)
    parser.add_argument("--moving-mm", type=float, default=5.)
    parser.add_argument("--history-points", type=int, default=3000, help="Bounded number of paired samples drawn")
    args = parser.parse_args(argv)
    for name in ("startup_timeout_s", "calibration_seconds", "signal_timeout", "max_gap", "stall_seconds", "frozen_mm", "moving_mm"):
        if name == "startup_timeout_s" and args.startup_timeout_s is None:
            continue
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not args.host or not args.mocap_topic or not 1 <= args.port <= 65535:
        parser.error("Require a host, mocap topic and UDP port in 1..65535")
    if args.history_points < 2 or (args.mocap_offset is not None and not np.isfinite(args.mocap_offset).all()):
        parser.error("History must have at least two points and offsets must be finite")
    if args.moving_mm <= args.frozen_mm:
        parser.error("--moving-mm must exceed --frozen-mm")
    if args.mocap_offset is not None:
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
        mocap_worker = partial(run_mocap_worker, wait_for_data=args.startup_timeout_s is None)
        for name, target in (("quest", run_quest_worker), ("mocap", mocap_worker)):
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
