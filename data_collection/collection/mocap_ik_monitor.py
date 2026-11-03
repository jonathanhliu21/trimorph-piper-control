"""Read-only, latest-sample Vicon IK feedback. Never connects to Piper."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import textwrap
import threading
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from data_collection.postprocess_dataset import vicon_flange_poses, vicon_to_piper
from data_collection.replay_actions import Limits, check_joints, piper_model
from .tracking_monitor import SIGNAL_TIMEOUT_S, stream_status
from .utils import import_cv2

MAX_RESIDUAL_MM, MAX_RESIDUAL_DEG = .5, .2
CHECK_PERIOD_S = .2
HISTORY_LEN = int(30 * 60 / CHECK_PERIOD_S)  # 30 min of checks; longer episodes lose their start.
PLOT_WINDOW_S = 10.  # Rolling window while not recording; minimum span while recording.
PLOT_GAP_S = 1.  # Break the trajectory line across longer tracking/IK gaps.
FINALIZE_DELAY_S = .5  # Let IK on a stopped episode's last samples arrive before scoring it.
DELETED_STOP_REASON = 'foot_pedal_delete'  # collect_hand_only.delete_episode; not a saved episode.
NO_IK_DETAIL = 'No accurate local IK within joint margins'
WIDTH, TEXT_HEIGHT, SCORE_HEIGHT, PLOT_ROWS, PLOT_COLS, PLOT_HEIGHT = 920, 330, 120, 2, 3, 200
PLOTS_TOP = TEXT_HEIGHT + SCORE_HEIGHT
GREEN, RED, AMBER, LIMIT = (60, 170, 60), (40, 40, 210), (0, 150, 200), (200, 200, 200)
PLOT_FILL, RED_FILL, RED_LINE = (40, 40, 40), (30, 30, 110), (90, 90, 255)


@dataclass(frozen=True)
class IKResult:
    sample_ns: int
    ok: bool
    detail: str
    joints_deg: tuple[float, ...] | None = None


class MocapIKChecker:
    """Bounded local IK seeded by the last valid solution, with replay pose checks."""

    def __init__(self, calibration, seed_deg, *, model=None):
        self.transform = vicon_to_piper(calibration)
        if any(not np.isfinite(value).all() for value in self.transform):
            raise ValueError('Vicon calibration contains nonfinite values')
        self.fk, self.bounds = piper_model() if model is None else model
        self.limits = Limits(joint_margin_deg=3., wrist_margin_deg=10.)
        margin = np.radians(self.limits.joint_margin_deg)
        self.lower, self.upper = self.bounds[:, 0] + margin, self.bounds[:, 1] - margin
        self.seed = np.radians(np.asarray(seed_deg, float))
        if self.seed.shape != (6,) or not np.isfinite(self.seed).all():
            raise ValueError('Mocap IK seed must contain six finite joint angles')

    def out_of_range(self, joints_deg):
        """Per joint: outside the margined limits, or J5 inside the wrist singularity band."""
        q = np.radians(np.asarray(joints_deg, float))
        out = (q < self.lower) | (q > self.upper)
        out[4] |= abs(q[4]) < np.radians(self.limits.wrist_margin_deg)
        return out

    def check(self, sample):
        if sample.lost:
            return IKResult(sample.monotonic_ns, False, 'OBJECT LOST')
        position = np.asarray(sample.position, float).reshape(1, 3)
        quaternion = np.asarray(sample.quaternion, float)
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < .5:
            return IKResult(sample.monotonic_ns, False, 'Invalid mocap pose')
        xyz, rotations = vicon_flange_poses(self.transform, position, Rotation.from_quat(quaternion[None, :]))
        target, rotation = xyz[0], rotations[0]

        def residual(q):
            pose = self.fk(q)
            return np.r_[pose[:3] - target, .2 * (
                Rotation.from_euler('xyz', pose[3:]) * rotation.inv()).as_rotvec()]

        def fit(seed, bounds=(-np.inf, np.inf)):
            result = least_squares(residual, seed, bounds=bounds,
                                   max_nfev=150, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            error = residual(result.x)
            mm, degrees = np.linalg.norm(error[:3]) * 1000, np.degrees(np.linalg.norm(error[3:]) / .2)
            return result.x, mm, degrees, result.success and mm <= MAX_RESIDUAL_MM and degrees <= MAX_RESIDUAL_DEG

        q, mm, degrees, accurate = fit(np.clip(self.seed, self.lower, self.upper), (self.lower, self.upper))
        if not accurate:
            detail = f'{NO_IK_DETAIL}: {mm:.2f} mm, {degrees:.2f} deg residual'
            # The bounded fit stops at the limits; refit unbounded to show how far past them the pose needs.
            q, _, _, accurate = fit(q)
            if not accurate:
                return IKResult(sample.monotonic_ns, False, detail)
            joints = tuple(np.degrees(q))
            outside = np.flatnonzero(self.out_of_range(joints))
            if len(outside):
                detail += '; needs ' + ', '.join(f'J{j+1}={joints[j]:.1f}' for j in outside) + ' deg'
            return IKResult(sample.monotonic_ns, False, detail, joints)
        joints = tuple(np.degrees(q))
        try:
            check_joints(q, self.bounds, self.fk, self.limits)
        except ValueError as exc:
            return IKResult(sample.monotonic_ns, False, str(exc), joints)
        self.seed = q.copy()
        return IKResult(sample.monotonic_ns, True, 'Local IK passes joint margins, wrist and Jacobian checks', joints)


class MocapIKMonitor:
    """OpenCV on the main thread; bounded IK on a separate latest-only worker."""

    def __init__(self, shared, calibration_path: Path, *, topic='/vicon/Yori/Yori',
                 seed_deg=(0., 45., -45., 0., 30., 0.), recorder=None, cv2=None):
        self.checker = MocapIKChecker(json.loads(Path(calibration_path).read_text()), seed_deg)
        self.shared, self.topic, self.recorder = shared, topic, recorder
        self.cv2 = cv2 or import_cv2()
        self.window_name = 'trimorph_mocap_ik'
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._result = None
        self._last_render_ns = None
        # (sample_ns, joints_deg or None, per-joint out-of-range flags or None, pose unreachable)
        self._history = deque(maxlen=HISTORY_LEN)
        self._episode, self._stopped = None, []
        self.episodes_in_range = self.episodes_scored = 0
        self.last_episode = None  # (episode index, in range, detail)
        self.cv2.namedWindow(self.window_name)
        self._thread = threading.Thread(target=self._run, name='mocap-ik-monitor', daemon=True)
        self._thread.start()

    def _run(self):
        last_count = None
        while not self._stop.is_set() and not self.shared.shutdown.is_set():
            sample = self.shared.latest_tracker_poses().get(self.topic)
            if sample is not None and sample.count != last_count:
                last_count = sample.count
                if stream_status(time.monotonic_ns(), sample, None, check_frozen=False).ok:
                    try:
                        result = self.checker.check(sample)
                    except Exception as exc:
                        result = IKResult(sample.monotonic_ns, False, f'IK check failed: {exc}')
                    self._record(result)
            self._stop.wait(CHECK_PERIOD_S)  # At most 5 checks/s; recording retains every raw sample.

    def _record(self, result):
        joints = result.joints_deg
        out = None if joints is None else self.checker.out_of_range(joints)
        unreachable = joints is None and result.detail.startswith(NO_IK_DETAIL)
        with self._lock:
            self._result = result
            self._history.append((result.sample_ns, joints, out, unreachable))

    def episode_verdict(self, start_ns, stop_ns):
        """In range when every IK result between start and stop had all joints in range.

        A pose with no joint solution at all counts against the episode; tracking loss does not.
        """
        with self._lock:
            entries = [entry for entry in self._history if start_ns <= entry[0] <= stop_ns]
        bad, unreachable = np.zeros(6, bool), 0
        for _, _, out, no_solution in entries:
            if out is not None:
                bad |= out
            unreachable += no_solution
        problems = [f'J{j+1}' for j in np.flatnonzero(bad)] + ([f'{unreachable} unreachable'] if unreachable else [])
        if problems:
            return False, 'out of range: ' + ', '.join(problems)
        if not any(entry[2] is not None for entry in entries):
            return False, 'no IK results (tracking lost?)'
        return True, 'all joints in range'

    def track_episodes(self, now_ns=None):
        """Score each saved episode once, after its last IK results arrive; deleted episodes do not count."""
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        session = self.recorder.current() if self.recorder is not None else None
        if session is not self._episode:
            if self._episode is not None:
                self._stopped.append(self._episode)
            self._episode = session
        for done in [episode for episode in self._stopped if episode.stop_ns is not None
                     and now_ns - episode.stop_ns >= FINALIZE_DELAY_S * 1e9]:
            self._stopped.remove(done)
            if done.stop_reason == DELETED_STOP_REASON:
                continue
            in_range, detail = self.episode_verdict(done.start_ns, done.stop_ns)
            self.episodes_scored += 1
            self.episodes_in_range += in_range
            self.last_episode = (done.paths.episode_index, in_range, detail)
            print(f'IK range score {self.episodes_in_range}/{self.episodes_scored}: '
                  f'episode {done.paths.episode_index:06d} {detail}')

    def status(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        sample = self.shared.latest_tracker_poses().get(self.topic)
        signal = stream_status(now_ns, sample, None, check_frozen=False)
        with self._lock:
            result = self._result
        if not signal.ok:
            return 'NO IK RESULT', signal.text, None
        if result is None:
            return 'CHECKING IK', 'Waiting for the first IK solution', None
        if (now_ns - result.sample_ns) / 1e9 > SIGNAL_TIMEOUT_S:
            return 'IK RESULT STALE', 'Waiting for IK on a recent pose', None
        return ('IK OK (local branch)' if result.ok else 'IK WARNING'), result.detail, result

    def plot_span(self, now_ns):
        """Episode since its start while recording, else the last PLOT_WINDOW_S."""
        session = self.recorder.current() if self.recorder is not None else None
        if session is None:
            return now_ns - int(PLOT_WINDOW_S * 1e9), now_ns, f'NOT RECORDING: last {PLOT_WINDOW_S:g} s'
        elapsed = (now_ns - session.start_ns) / 1e9
        return (session.start_ns, max(now_ns, session.start_ns + int(PLOT_WINDOW_S * 1e9)),
                f'REC {elapsed:.1f} s: whole episode')

    def render(self, now_ns=None):
        cv2 = self.cv2
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        title, detail, result = self.status(now_ns)
        t0, t1, span_text = self.plot_span(now_ns)
        color = GREEN if result is not None and result.ok else RED if result is not None else AMBER
        image = np.full((PLOTS_TOP + PLOT_ROWS * PLOT_HEIGHT, WIDTH, 3), 28, dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (WIDTH, 62), color, -1)
        lines = [title, self.topic, *textwrap.wrap(detail, width=95, max_lines=2, placeholder=' ...')]
        if result is not None and result.joints_deg is not None:
            lines += ['IK joints (deg):', '  '.join(
                f'J{i+1}: {value:.1f}' for i, value in enumerate(result.joints_deg))]
        lines += ['Absolute calibrated flange pose; SDK joint limits, 3 deg margin, |J5| >= 10 deg.',
                  'Pose check only: no path, collision, controller-limit or training-distribution check.',
                  'Local IK failure does not prove all joint branches are infeasible.',
                  f'Plots ({span_text}): dotted = limits incl. margin; red = out of range.']
        for i, line in enumerate(lines):
            cv2.putText(image, line, (14, 38 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                        .52, (255, 255, 255), 1, cv2.LINE_AA)
        self._draw_score(image, now_ns)
        with self._lock:
            history = [entry for entry in self._history if t0 <= entry[0] <= t1]
        width = WIDTH // PLOT_COLS
        for j in range(6):
            row, col = divmod(j, PLOT_COLS)
            self._plot_joint(image, j, history, t0, t1,
                             (col * width, PLOTS_TOP + row * PLOT_HEIGHT, width, PLOT_HEIGHT))
        return image

    def _draw_score(self, image, now_ns):
        cv2, top = self.cv2, TEXT_HEIGHT
        cv2.rectangle(image, (6, top), (WIDTH - 6, top + SCORE_HEIGHT - 6), PLOT_FILL, -1)
        score = f'{self.episodes_in_range}/{self.episodes_scored}'
        cv2.putText(image, score, (20, top + 78), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (255, 255, 255), 5, cv2.LINE_AA)
        (score_w, _), _ = cv2.getTextSize(score, cv2.FONT_HERSHEY_SIMPLEX, 2.6, 5)
        cv2.putText(image, 'episodes with all joints in range (this session)', (20, top + 104),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, LIMIT, 1, cv2.LINE_AA)
        x = max(380, 40 + score_w)
        lines = []
        if self._episode is not None:
            in_range, detail = self.episode_verdict(self._episode.start_ns, now_ns)
            if in_range or not detail.startswith('no IK'):
                lines.append((f'Recording {self._episode.paths.episode_index:06d}: {detail}',
                              GREEN if in_range else RED_LINE))
        if self.last_episode is not None:
            index, in_range, detail = self.last_episode
            lines.append((f'Last saved {index:06d}: ' + ('IN RANGE' if in_range else detail),
                          GREEN if in_range else RED_LINE))
        for i, (line, color) in enumerate(lines):
            scale = .6
            while scale > .35 and cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > WIDTH - 20 - x:
                scale -= .05
            cv2.putText(image, line, (x, top + 42 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def _plot_joint(self, image, j, history, t0, t1, rect):
        cv2 = self.cv2
        x0, y0, w, h = rect
        x0, y0, w, h = x0 + 6, y0 + 6, w - 12, h - 12
        times = np.array([entry[0] for entry in history], float)
        values = np.array([np.nan if entry[1] is None else entry[1][j] for entry in history])
        out = np.array([entry[2] is not None and entry[2][j] for entry in history], bool)
        last = np.flatnonzero(np.isfinite(values))
        # The fill follows the newest solution; earlier excursions stay visible as red trace.
        cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h),
                      RED_FILL if len(last) and out[last[-1]] else PLOT_FILL, -1)
        limits = [np.degrees(self.checker.lower[j]), np.degrees(self.checker.upper[j])]
        if j == 4:
            limits += [-self.checker.limits.wrist_margin_deg, self.checker.limits.wrist_margin_deg]
        finite = values[np.isfinite(values)]
        low, high = min(*limits, *finite), max(*limits, *finite)
        low, high = low - .08 * (high - low), high + .08 * (high - low)
        top, plot_h = y0 + 22, h - 26  # Leave room for the joint label.

        def to_y(value):
            return np.round(top + (high - value) / (high - low) * plot_h).astype(np.int32)

        for limit in limits:
            y = int(to_y(limit))
            for x in range(x0, x0 + w, 10):
                cv2.line(image, (x, y), (min(x + 5, x0 + w), y), LIMIT, 1)
        xs = np.round(x0 + (times - t0) / (t1 - t0) * w).astype(np.int32)
        points = np.stack([xs, to_y(np.nan_to_num(values))], axis=1)
        if len(history) > 1:
            valid = np.isfinite(values[:-1]) & np.isfinite(values[1:]) & (np.diff(times) <= PLOT_GAP_S * 1e9)
            bad = out[:-1] | out[1:]
            for segment_color, mask in ((GREEN, valid & ~bad), (RED_LINE, valid & bad)):
                index = np.flatnonzero(mask)
                if len(index):
                    cv2.polylines(image, list(np.stack([points[index], points[index + 1]], axis=1)),
                                  False, segment_color, 2, cv2.LINE_AA)
        if len(last):  # Mark the newest solution, which may have no neighbour to draw a line to.
            i = last[-1]
            cv2.circle(image, tuple(int(v) for v in points[i]),
                       4, RED_LINE if out[i] else GREEN, -1, cv2.LINE_AA)
        label = f'J{j+1}' + (f'  {finite[-1]:.1f} deg' if len(finite) else '')
        cv2.putText(image, label, (x0 + 6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255), 1, cv2.LINE_AA)

    def update(self):
        now_ns = time.monotonic_ns()
        self.track_episodes(now_ns)  # Every tick, so a quick stop/start cannot hide an episode.
        # Results arrive at most every CHECK_PERIOD_S, so redraw no faster (a long episode takes ~20 ms).
        if self._last_render_ns is None or now_ns - self._last_render_ns >= CHECK_PERIOD_S * 1e9:
            self._last_render_ns = now_ns
            self.cv2.imshow(self.window_name, self.render(now_ns))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.)
        self.cv2.destroyWindow(self.window_name)
