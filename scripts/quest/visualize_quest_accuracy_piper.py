#!/usr/bin/env python3
"""Passive, synchronized Quest flange versus measured Piper FK comparison.

    python scripts/quest/visualize_quest_accuracy_piper.py --quest-port 5005
    python scripts/quest/visualize_quest_accuracy_piper.py --calibration CAL.json

Rigidly attach Quest to the flange. Move with Piper's built-in drag-teach mode;
this program never enables the arm or sends motion commands. C captures a live
calibration: translate in 3D and independently rotate about multiple axes. Or
load fit_quest_calibration.py's JSON. The mapping is anchored at the first paired
pose, so this measures relative tracking agreement, not surveyed absolute error.
A re-anchors without refitting; R clears statistics. Drag rotates, scroll zooms.
Orange X marks flag raw Quest position steps above --jump-threshold (meters),
at the mapped jump destination. Markers persist with the trajectory history.
Q/Esc exits. Live fits use rigid+reflection+offset, with independent orientation
calibration, matching get_coord_transform.py. Training errors are not validation.

Quest received at t is paired with interpolated Piper FK at t-latency_s, matching
the fitter's convention. No extrapolation or interpolation across long gaps.
Host receive timestamps cannot eliminate variable network/sensor latency.
IK uses the collection monitor's bounded local solver, joint margins, wrist and
Jacobian checks on the Quest-derived target, in a background thread. Local IK
failure does not prove every branch infeasible; it is not a collision check.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_collection.collection import config as defaults
from data_collection.collection.mocap_ik_monitor import IKResult, MocapIKChecker
from data_collection.collection.piper_worker import piper_workspace_warning
from data_collection.collection.quest_worker import QUEST_TRACKER_NAME, run_quest_worker
from data_collection.collection.shared_state import TrackerSample
from data_collection.collection.tracking_monitor import AXIS_COLORS, QUEST_COLOR, VICON_COLOR, TrackingMonitor, project
from data_collection.collection.utils import worker_wrapper
from scripts.calib.get_coord_transform import fit_pose_transform, transform_chunk, tracked_point_offset
from scripts.quest.visualize_quest_accuracy import NoRecording, PoseHistory, frozen_status, signal_status, synchronized_pose

PIPER = 'piper_fk'
RED, GREEN, GRAY, AMBER = (60, 60, 255), (80, 220, 80), (195, 195, 195), (0, 180, 255)


def validate_calibration(value):
    a = np.asarray(value.get('linear_matrix'), float)
    b = np.asarray(value.get('orientation', {}).get('rotation_matrix'), float)
    if a.shape != (3, 3) or not np.isfinite(a).all() or np.linalg.matrix_rank(a) < 3:
        raise ValueError('Calibration needs a finite, nonsingular linear_matrix')
    if (b.shape != (3, 3) or not np.isfinite(b).all()
            or not np.allclose(b.T @ b, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(b), 1., atol=1e-6)):
        raise ValueError('Calibration needs a proper orientation.rotation_matrix')
    if not math.isfinite(float(value.get('latency_s', 0.))):
        raise ValueError('Calibration latency_s must be finite')
    if value.get('tracker', 'quest') != 'quest':
        raise ValueError('Use a Quest calibration, not a mocap tracker calibration')
    tracked_point_offset(value)
    return value


@dataclass(frozen=True)
class WorkerConfig:
    quest_host: str
    quest_port: int
    quest_startup_timeout_s: float | None = None


class PiperHistory(PoseHistory):
    def __init__(self):
        super().__init__()
        self.joints = None
        self.arm_state = {}

    def set_feedback(self, joints, state):
        with self._lock:
            self.joints = tuple(joints)
            self.arm_state = dict(state)

    def feedback(self):
        with self._lock:
            return self.joints, dict(self.arm_state)


def run_piper(args, shared, *, init_done):
    from scripts.piper_api.piper_api import PiperAPI
    # None leaves installation/payload settings untouched on this passive connection.
    api = PiperAPI(firmware=args.firmware, interface=args.interface, channel=args.channel,
                   installation_pos=None, payload=None)
    try:
        if not api.connect():
            raise RuntimeError('Piper is not responding on the CAN bus')
        init_done.set()
        previous_stamp = None
        while not shared.shutdown.is_set():
            joints = api.get_joint_state()
            if joints is not None and joints.timestamp != previous_stamp:
                previous_stamp = joints.timestamp
                pose = np.asarray(api.robot.fk(list(joints.joints)), float)
                if pose.shape != (6,) or not np.isfinite(pose).all():
                    raise ValueError('Invalid Piper joint-feedback FK')
                shared.set_tracker_pose(PIPER, time.monotonic_ns(), tuple(pose[:3]),
                                        tuple(Rotation.from_euler('xyz', pose[3:]).as_quat()))
                shared.set_feedback(joints.joints, api.get_state())
            shared.shutdown.wait(1. / args.piper_rate_hz)
    finally:
        api.disconnect()


class LatestIK:
    """At most five checks/sec; replace pending targets rather than queueing them."""
    def __init__(self, shared):
        self.shared = shared
        self.lock = threading.Lock()
        self.pending = self.result = None
        self.thread = threading.Thread(target=self.run, name='quest-piper-ik', daemon=True)
        self.thread.start()

    def submit(self, sample, joints, generation):
        with self.lock:
            self.pending = sample, joints, generation

    def snapshot(self):
        with self.lock:
            return self.result

    def run(self):
        checker = None
        while not self.shared.shutdown.is_set():
            with self.lock:
                pending, self.pending = self.pending, None
            if pending is not None:
                sample, joints, generation = pending
                try:
                    if checker is None:
                        checker = MocapIKChecker(None, np.degrees(joints))
                    # Current measured joints select the nearby physical arm branch.
                    checker.seed = np.asarray(joints, float)
                    result = checker.check(sample)
                except Exception as exc:
                    result = IKResult(sample.monotonic_ns, False, f'IK check failed: {exc}')
                with self.lock:
                    self.result = generation, result
            self.shared.shutdown.wait(.2)


class Comparison:
    """Timestamp pairing, fixed-anchor mapping and statistics, without GUI/hardware."""
    def __init__(self, args, calibration=None):
        self.args = args
        self.calibration = None if calibration is None else validate_calibration(calibration)
        self.latency = args.latency_s if args.latency_s is not None else float((calibration or {}).get('latency_s', 0.))
        self.last_stamp = -1
        self.anchor = None
        self.generation = 0
        self.capture_end = None
        self.capture = []
        self.last_capture = -1
        self.message = 'Press C to calibrate; rigidly attach Quest to Piper flange'
        self.note = 'Waiting for paired poses'
        self.clear()

    def clear(self):
        self.trail = deque(maxlen=1500)
        self.count = 0
        self.squares = np.zeros(2)
        self.maximum = np.zeros(2)
        self.latest = None
        self.recent = deque(maxlen=1000)
        self.stall = ''
        self.previous_quest = None
        self.jumps = deque(maxlen=1500)

    def reanchor(self):
        self.anchor = None
        self.generation += 1
        self.clear()
        self.message = 'Waiting for a fresh synchronized anchor'

    def start_capture(self, now):
        self.capture_end = now + round(self.args.calibration_seconds * 1e9)
        self.capture = []
        self.last_capture = -1
        self.reanchor()
        self.message = 'CALIBRATING: translate in THREE directions, rotate about TWO axes'

    def ingest(self, history, now):
        targets = []
        piper = history.get(PIPER, [])
        for sample in history.get(QUEST_TRACKER_NAME, []):
            if sample.monotonic_ns <= self.last_stamp:
                continue
            stamp = sample.monotonic_ns - round(self.latency * 1e9)
            if not piper or stamp > piper[-1].monotonic_ns:
                self.note = 'Waiting for newer Piper FK to bracket Quest timestamp'
                if now - sample.monotonic_ns < self.args.signal_timeout * 1e9:
                    break
                self.last_stamp = sample.monotonic_ns
                continue
            self.last_stamp = sample.monotonic_ns
            pose = synchronized_pose(piper, stamp, self.args.max_gap)
            if pose is None or sample.lost:
                self.note = 'No valid Piper interpolation bracket (gap/lost/old sample)'
                continue
            q = np.asarray(sample.position)
            quat = np.asarray(sample.quaternion)
            if not np.isfinite(q).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) < .5:
                self.note = 'Invalid Quest pose'
                continue
            p, r = pose
            self.recent.append((sample.monotonic_ns, q, p))
            while self.recent and sample.monotonic_ns - self.recent[0][0] > 1e9:
                self.recent.popleft()
            self.stall = frozen_status(self.recent, 1., .0001, .005).replace('MOCAP', 'PIPER')
            if self.capture_end is not None:
                # Decimate capture to <=50 Hz; bounded by calibration duration.
                if (sample.monotonic_ns <= self.capture_end
                        and sample.monotonic_ns - self.last_capture >= 20_000_000):
                    self.capture.append((q, quat, p, r))
                    self.last_capture = sample.monotonic_ns
                continue
            if self.calibration is None:
                continue
            if self.anchor is None:
                self.anchor = (q.copy(), quat.copy(), p.copy(), r.copy())
            q0, quat0, p0, r0 = self.anchor
            mapped, mapped_quat = transform_chunk([q0, q], [quat0, quat], p0, Rotation.from_matrix(r0).as_quat(), self.calibration)
            target, rotation = mapped[-1], Rotation.from_quat(mapped_quat[-1]).as_matrix()
            errors = np.array([1000 * np.linalg.norm(target - p),
                               np.degrees(Rotation.from_matrix(rotation @ r.T).magnitude())])
            self.count += 1
            self.squares += errors ** 2
            self.maximum = np.maximum(self.maximum, errors)
            self.latest = (sample.monotonic_ns, target, rotation, p, r, errors)
            self.trail.append(self.latest)
            if self.previous_quest is not None:
                previous_stamp, previous_position = self.previous_quest
                if (sample.monotonic_ns - previous_stamp <= self.args.max_gap * 1e9
                        and np.linalg.norm(q - previous_position) > self.args.jump_threshold):
                    self.jumps.append((sample.monotonic_ns, target.copy()))
            self.previous_quest = sample.monotonic_ns, q.copy()
            while self.jumps and self.jumps[0][0] < self.trail[0][0]:
                self.jumps.popleft()
            self.note = 'Paired successfully'
            targets.append(TrackerSample(sample.monotonic_ns, tuple(target), tuple(mapped_quat[-1]), False, sample.count))
        return targets


def fit_capture(pairs, latency):
    if len(pairs) < 20:
        raise ValueError('Need at least 20 paired poses; check both streams')
    q, quat, p, r = (np.asarray(v) for v in zip(*pairs))
    result = fit_pose_transform(q, p, quat, Rotation.from_matrix(r).as_quat(),
                                model='rigid', allow_reflection=True, fit_offset=True)
    result.update(latency_s=latency, tracker='quest', selected_model='rigid+refl+offset')
    return validate_calibration(result)


class PiperView(TrackingMonitor):
    def __init__(self, shared, args, comparison):
        super().__init__(shared, show_quest=False, show_vicon=False,
                         window_name='quest_accuracy_piper', size=(1280, 900))
        self.args, self.comparison = args, comparison

    def draw(self, history, now, ik):
        cv2, c = self.cv2, self.comparison
        replay = getattr(self, 'replay', False)
        image = np.full((self.height, self.width, 3), 28, np.uint8)

        def text(value, y, color=GRAY, scale=.53):
            # Wrap long solver diagnostics to keep the entire reason visible.
            import textwrap
            for i, line in enumerate(textwrap.wrap(value, 145) or ['']):
                cv2.putText(image, line, (16, y + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

        statuses = [signal_status(history.get(name, [None])[-1], now, self.args.signal_timeout)
                    for name in (QUEST_TRACKER_NAME, PIPER)]
        reference_label = 'RECORDED PIPER' if replay else 'QUEST'
        text(f'{reference_label}: {statuses[0]}     LIVE PIPER FK: {statuses[1]}', 28,
             GREEN if statuses == ['OK', 'OK'] else RED)
        fresh = (c.latest is not None and all(s == 'OK' for s in statuses)
                 and now - c.latest[0] < self.args.signal_timeout * 1e9)
        if replay and getattr(c, 'replay_phase', 'waiting') != 'trajectory':
            fresh = False
        if c.capture_end is not None:
            text(f'CALIBRATING {max(0., (c.capture_end-now)/1e9):.1f}s | paired poses: {len(c.capture)}', 58, AMBER)
        elif fresh:
            mm, deg = c.latest[-1]
            comparison_label = 'Executed FK vs recorded Piper' if replay else 'Quest flange vs Piper FK'
            text(f'{comparison_label}: {mm:.2f} mm   {deg:.2f} deg', 58,
                 RED if mm > self.args.position_threshold_mm or deg > self.args.orientation_threshold_deg else GREEN, .7)
        else:
            reason = ('calibration required' if c.calibration is None else
                      'stream stale or lost' if any(s != 'OK' for s in statuses) else c.note)
            text('ERROR UNAVAILABLE: ' + reason, 58, AMBER)
        if c.count:
            rms = np.sqrt(c.squares / c.count)
            text(f'N={c.count} | RMSE {rms[0]:.2f} mm / {rms[1]:.2f} deg | max {c.maximum[0]:.2f} mm / {c.maximum[1]:.2f} deg', 88)
        text(c.message, 118, AMBER)
        text('Source-time aligned | orange=original recorded Piper, cyan=live executed Piper FK' if replay else
             f'Relative anchored comparison | latency {c.latency*1000:+.1f} ms | orange=Quest flange, cyan=Piper FK', 159)
        joints, state = self.shared.feedback()
        warning = piper_workspace_warning(state)
        text(warning or 'Piper firmware: no reported kinematic limit', 188, RED if warning else GRAY)
        result = None if ik is None or ik[0] != c.generation else ik[1]
        if not fresh or result is None or now - result.sample_ns > 1e9:
            text('IK: waiting for fresh mapped target/result', 232, AMBER)
        else:
            text(('IK OK: ' if result.ok else 'IK WARNING: ') + result.detail, 232, GREEN if result.ok else RED)
            if result.joints_deg is not None:
                text('IK joints (deg): ' + '  '.join(f'J{i+1}={v:.1f}' for i, v in enumerate(result.joints_deg)),
                     292, GREEN if result.ok else RED)
        if joints is not None:
            text('Measured (deg): ' + '  '.join(f'J{i+1}={v:.1f}' for i, v in enumerate(np.degrees(joints))), 317)
        text('Replay executes the Quest-derived plan; displayed error compares live FK with original recorded Piper.' if replay else
             'Local IK failure does not rule out other branches. No collision check. Passive: no arm commands.', 342, AMBER)
        if fresh and c.stall:
            text(c.stall, 365, RED)
        text('Approach excluded from comparison statistics; no per-frame alignment or error subtraction.' if replay else
             f'Orange X: Quest position jump > {self.args.jump_threshold*1000:g} mm', 390, QUEST_COLOR)
        origin = c.anchor[2] if c.anchor is not None else np.zeros(3)

        def screen(points):
            return project(np.asarray(points) - origin, self.yaw, self.pitch, self.scale, (640, 570))

        for v in np.arange(-.5, .51, .1):
            for points in ([[-.5, v, 0], [.5, v, 0]], [[v, -.5, 0], [v, .5, 0]]):
                a, b = screen(np.asarray(points) + origin)
                cv2.line(image, tuple(a), tuple(b), (50, 50, 50), 1)
        previous = None
        for stamp, q, rq, p, rp, error in c.trail:
            a, b = screen([q, p])
            bad = error[0] > self.args.position_threshold_mm or error[1] > self.args.orientation_threshold_deg
            if previous is not None and stamp - previous[0] <= self.args.max_gap * 1e9:
                cv2.line(image, tuple(previous[1]), tuple(a), QUEST_COLOR, 1)
                cv2.line(image, tuple(previous[2]), tuple(b), VICON_COLOR, 1)
            if bad:
                cv2.line(image, tuple(a), tuple(b), RED, 1)
            previous = stamp, a, b
        if c.latest is not None:
            for p, r, color in ((c.latest[1], c.latest[2], QUEST_COLOR), (c.latest[3], c.latest[4], VICON_COLOR)):
                start = screen([p])[0]
                if color == QUEST_COLOR and result is not None and not result.ok and now - result.sample_ns <= 1e9:
                    color = RED
                cv2.circle(image, tuple(start), 6, color if fresh else GRAY, -1)
                for axis, axis_color in zip(r.T, AXIS_COLORS):
                    end = screen([p + self.args.axis_length * axis])[0]
                    cv2.line(image, tuple(start), tuple(end), axis_color if fresh else GRAY, 2)
        for _, position in c.jumps:
            x, y = screen([position])[0]
            cv2.line(image, (int(x)-7, int(y)-7), (int(x)+7, int(y)+7), QUEST_COLOR, 2, cv2.LINE_AA)
            cv2.line(image, (int(x)-7, int(y)+7), (int(x)+7, int(y)-7), QUEST_COLOR, 2, cv2.LINE_AA)
        text('C: capture calibration | A: re-anchor | R: reset statistics | drag: rotate | scroll/right-drag: zoom | Q/Esc: quit', 860)
        text('Errors include both trackers, FK/model error, mount flex and timing. Calibration is fixed during measurement.', 885)
        return image


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--save-calibration', type=Path, help='Save successful live C fits to this JSON path')
    parser.add_argument('--quest-host', default=defaults.DEFAULT_QUEST_HOST)
    parser.add_argument('--quest-port', type=int, default=defaults.DEFAULT_QUEST_PORT)
    parser.add_argument('--channel', default=defaults.DEFAULT_PIPER_CHANNEL)
    parser.add_argument('--interface', default=defaults.DEFAULT_PIPER_INTERFACE)
    parser.add_argument('--firmware', default=defaults.DEFAULT_PIPER_FIRMWARE)
    parser.add_argument('--latency-s', type=float, help='Override saved latency; Quest(t+latency) matches Piper(t)')
    for flag, default in (('calibration-seconds', 20.), ('piper-rate-hz', 200.), ('max-gap', .05),
                          ('signal-timeout', .5), ('position-threshold-mm', 10.),
                          ('orientation-threshold-deg', 5.), ('axis-length', .06), ('jump-threshold', .05)):
        parser.add_argument('--' + flag, type=float, default=default)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, float) and (not math.isfinite(value) or (name != 'latency_s' and value <= 0)):
            parser.error(f'--{name.replace("_", "-")} must be finite' + (' and positive' if name != 'latency_s' else ''))
    if not 1 <= args.quest_port <= 65535:
        parser.error('--quest-port must be in 1..65535')
    return args


def main(argv=None):
    args = parse_args(argv)
    calibration = None if args.calibration is None else json.loads(args.calibration.read_text())
    comparison, shared = Comparison(args, calibration), PiperHistory()
    view = PiperView(shared, args, comparison)
    errors = queue.Queue()
    threads = []
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='quest-piper-fit')
    pending_fit = None
    ik = LatestIK(shared)
    try:
        for name, target, config, extra in (
                ('piper', run_piper, args, ()),
                ('quest', run_quest_worker, WorkerConfig(args.quest_host, args.quest_port), (NoRecording(),))):
            thread = threading.Thread(target=worker_wrapper,
                                      args=(name, errors, shared, threading.Event(), target, config, shared, *extra), daemon=True)
            threads.append(thread)
            thread.start()
        print('Passive Piper/Quest viewer. Attach Quest rigidly; use built-in drag-teach. C calibrates; Q exits.')
        while not shared.shutdown.is_set():
            now = time.monotonic_ns()
            history = shared.snapshot()
            targets = comparison.ingest(history, now)
            joints, _ = shared.feedback()
            if targets and joints is not None:
                ik.submit(targets[-1], joints, comparison.generation)
            if comparison.capture_end is not None and now >= comparison.capture_end:
                comparison.capture_end = None
                pending_fit = executor.submit(fit_capture, comparison.capture, comparison.latency)
                comparison.message = 'Fitting calibration in background...'
            if pending_fit is not None and pending_fit.done():
                try:
                    fitted = pending_fit.result()
                    comparison.calibration = fitted
                    comparison.reanchor()
                    fit_errors = fitted['fit_errors']
                    comparison.message = (f'Fit training RMS {1000*fit_errors["position_rmse"]:.2f} mm / '
                                          f'{np.degrees(fitted["orientation"]["fit_rmse_rad"]):.2f} deg; validate on NEW poses')
                    if args.save_calibration is not None:
                        args.save_calibration.parent.mkdir(parents=True, exist_ok=True)
                        args.save_calibration.write_text(json.dumps(fitted, indent=2, allow_nan=False) + '\n')
                    print(comparison.message, flush=True)
                except (ValueError, np.linalg.LinAlgError, OSError) as exc:
                    comparison.message = f'Calibration failed: {exc}'
                    print(comparison.message, flush=True)
                pending_fit = None
            view.cv2.imshow(view.window_name, view.draw(history, now, ik.snapshot()))
            key = view.cv2.waitKeyEx(20)
            if key in (27, ord('q'), ord('Q')) or view.cv2.getWindowProperty(view.window_name, view.cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('c'), ord('C')) and pending_fit is None:
                comparison.start_capture(now)
            elif key in (ord('a'), ord('A')):
                comparison.reanchor()
            elif key in (ord('r'), ord('R')):
                comparison.clear()
        if not errors.empty():
            error = errors.get_nowait()
            raise RuntimeError(f'{error.name}: {error.message}\n{error.traceback_text}')
    except KeyboardInterrupt:
        return 130
    finally:
        shared.shutdown.set()
        for thread in threads:
            thread.join(timeout=2.)
        ik.thread.join(timeout=2.)
        executor.shutdown(wait=True, cancel_futures=True)
        view.close()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
