#!/usr/bin/env python3
"""Plan/replay processed Quest-derived arm actions and log comparison to original FK.

Uses data_collection.replay_actions, including IK, controller limits, runtime
feedback checks and smooth joint replay. Planning is the default; --execute
explicitly enables motion. The same accuracy viewer opens during execution.
No Quest UDP receiver or hand is used during replay.
Configured motion and proximity thresholds warn by default; --strict-limits
makes them errors again. Controller bounds, failed IK and unusable feedback still stop.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_collection import replay_actions
from data_collection.collection.quest_worker import QUEST_TRACKER_NAME
from scripts.quest.visualize_quest_accuracy_piper import Comparison, PiperHistory, PIPER, parse_args as view_args
from scripts.quest_piper.common import errors, metrics, write_csv, write_json


class ReplayComparison:
    def __init__(self, original, stamps, speed, output, *, show=True, warn_margins=False):
        self.original, self.stamps, self.speed = original, stamps, speed
        self.times = (stamps-stamps[0])/1e9
        self.rotations = Rotation.from_rotvec(original[:, 3:])
        self.slerp = Slerp(self.times, self.rotations)
        self.output, self.rows, self.view = output, [], None
        self.show = show
        self.warn_margins = warn_margins
        self.shared = PiperHistory()
        self.comparison = Comparison(view_args([]), dict(linear_matrix=np.eye(3).tolist(),
                                   orientation={'rotation_matrix': np.eye(3).tolist()}))
        self.comparison.message = 'REPLAY: orange=original recorded Piper, cyan=live executed FK; Q/Esc stops'
        self.comparison.replay_phase = 'waiting'
        self.last_draw = 0.
        self.fk = None

    def prepare(self, plan, fk):
        self.fk = fk
        if self.show:
            from scripts.quest_piper.visualizer import PipelineView
            self.view = PipelineView(self.shared, self.comparison.args, self.comparison)
            self.view.replay = True
            self.view.original = self.original[:, :3]
            self.view.cv2.imshow(self.view.window_name, self.view.draw({}, time.monotonic_ns(), None))
            self.view.cv2.waitKeyEx(1)

    def __call__(self, plan, index, elapsed, stamp, measured):
        if self.fk is None:
            self.fk, _ = replay_actions.piper_model()
        measured_pose = self.fk(measured)
        p, r = measured_pose[:3], Rotation.from_euler('xyz', measured_pose[3:])
        target = plan['poses'][index]
        # Preposition prepends time; the source trajectory keeps its original duration/speed.
        approach = max(0., float(plan['times'][-1])-self.times[-1]/self.speed)
        schedule_delay = float(plan.get('runtime_delay_s', 0.))
        source_time = (elapsed-schedule_delay-approach)*self.speed
        in_trajectory = 0 <= source_time <= self.times[-1]
        clipped = float(np.clip(source_time, 0., self.times[-1]))
        original_p = np.array([np.interp(clipped, self.times, self.original[:, j]) for j in range(3)])
        original_r = self.slerp([clipped])
        original_error = errors(p[None, :], Rotation.from_quat(r.as_quat()[None, :]), original_p[None, :], original_r)[0]
        target_error = errors(p[None, :], Rotation.from_quat(r.as_quat()[None, :]), target[None, :3], Rotation.from_rotvec(target[None, 3:]))[0]
        row = dict(monotonic_ns=stamp, plan_index=index, scheduled_s=float(plan['times'][index]),
                   actual_replay_s=elapsed, schedule_delay_s=schedule_delay, source_elapsed_s=source_time,
                   phase='trajectory' if in_trajectory else 'approach' if source_time < 0 else 'after_source_end',
                   target_position_error_mm=float(target_error[0]), target_orientation_error_deg=float(target_error[1]),
                   original_position_error_mm=float(original_error[0]), original_orientation_error_deg=float(original_error[1]))
        row.update({f'measured_{key}': float(value) for key, value in zip(('x','y','z','roll','pitch','yaw'), measured_pose)})
        row.update({f'joint_{i+1}': float(value) for i, value in enumerate(measured)})
        row.update({f'target_{key}': float(value) for key, value in zip(('x','y','z','rx','ry','rz'), target)})
        row.update({f'original_{key}': float(value) for key, value in
                    zip(('x','y','z','rx','ry','rz'), np.r_[original_p, original_r.as_rotvec()[0]])})
        self.rows.append(row)
        c = self.comparison
        if c.anchor is None:
            c.anchor = (original_p, original_r.as_quat()[0], self.original[0, :3], self.rotations[0].as_matrix())
        c.latest = (stamp, original_p, original_r.as_matrix()[0], p, r.as_matrix(), original_error)
        c.replay_phase = row['phase']
        if in_trajectory:
            c.trail.append(c.latest)
            c.count += 1; c.squares += original_error**2; c.maximum = np.maximum(c.maximum, original_error)
        c.note = ('Live executed FK vs original recorded Piper at source time' if in_trajectory else
                  'Prepositioning: excluded from trajectory errors' if source_time < 0 else
                  'Source trajectory ended: excluded from trajectory errors')
        # Reuse the reference display slot; this is recorded Piper FK, not live Quest data.
        self.shared.set_tracker_pose(QUEST_TRACKER_NAME, stamp, tuple(original_p), tuple(original_r.as_quat()[0]))
        self.shared.set_tracker_pose(PIPER, stamp, tuple(p), tuple(r.as_quat()))
        self.shared.set_feedback(measured, {})
        if self.show and elapsed-self.last_draw >= .066:
            if self.view is None:
                from scripts.quest_piper.visualizer import PipelineView
                self.view = PipelineView(self.shared, c.args, c)
                self.view.replay = True
                self.view.original = self.original[:, :3]
            from data_collection.collection.mocap_ik_monitor import IKResult
            # IK is prevalidated by the replay planner; runtime guards run before this callback.
            detail = ('Replay IK solved; software thresholds advisory (see terminal)' if self.warn_margins else
                      'Replay IK prevalidated; runtime feedback checks passed')
            ik = (c.generation, IKResult(stamp, True, detail, tuple(np.degrees(plan['joints'][index]))))
            image = self.view.draw(self.shared.snapshot(), time.monotonic_ns(), ik)
            self.view.cv2.imshow(self.view.window_name, image)
            key = self.view.cv2.waitKeyEx(1)
            if key in (27, ord('q'), ord('Q')) or self.view.cv2.getWindowProperty(self.view.window_name, self.view.cv2.WND_PROP_VISIBLE) < 1:
                raise KeyboardInterrupt('Replay stopped from viewer')
            self.last_draw = elapsed

    def finish(self, completed):
        if self.view is not None:
            self.view.close()
        write_csv(self.output/'feedback.csv', self.rows)
        trajectory = [row for row in self.rows if row['phase'] == 'trajectory']
        def columns(prefix):
            return np.asarray([[row[f'{prefix}_position_error_mm'], row[f'{prefix}_orientation_error_deg']] for row in trajectory])
        write_json(self.output/'comparison.json', dict(completed=completed, samples=len(self.rows), warn_margins=self.warn_margins,
            primary_comparison='live_executed_piper_fk_vs_original_recorded_piper_fk',
            trajectory_vs_original=metrics(columns('original')), trajectory_vs_scheduled_target=metrics(columns('target')),
            note='Feedback sampled immediately before each command; errors include servo lag. Source time excludes approach and logged schedule delays.'))
        if trajectory:
            from scripts.quest_piper.postprocess import plot
            times = np.array([row['monotonic_ns'] for row in trajectory], dtype=np.int64)
            measured = np.array([[row[f'measured_{key}'] for key in ('x','y','z')] for row in trajectory])
            source_times = np.array([row['source_elapsed_s'] for row in trajectory])
            original = np.column_stack([np.interp(source_times, self.times, self.original[:, j]) for j in range(3)])
            plot(self.output/'comparison.png', times, columns('original'), measured, original, label='replayed FK')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--no-view', action='store_true')
    parser.add_argument('--preposition', action='store_true')
    parser.add_argument('--strict-limits', action='store_true',
                        help='Restore errors for configured motion/proximity thresholds instead of warnings')
    parser.add_argument('--max-joint-speed-deg-s', type=float, default=replay_actions.Limits().max_joint_speed_deg_s)
    parser.add_argument('--start-joints-deg', type=float, nargs=6, help='Offline planning seed; execute still reads actual arm')
    parser.add_argument('--playback-speed', type=float, default=.2)
    parser.add_argument('--rate-hz', type=float, default=30.)
    parser.add_argument('--piper-channel', default='can0')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if not np.isfinite(args.playback_speed) or not 0 < args.playback_speed <= 1:
        parser.error('--playback-speed must be in (0,1]')
    stem = f'{args.episode:06d}'
    actions = args.dataset/'absolute_action_t'/f'{stem}.npy'
    original = np.load(args.dataset/'original_action_t'/f'{stem}.npy', allow_pickle=False)
    targets = np.load(actions, allow_pickle=False)
    with np.load(args.dataset/'syncs'/f'{stem}.npz', allow_pickle=False) as sync:
        stamps = sync['pose_monotonic_ns'].copy()
    replay_actions.finite_array(original, 6, 'original poses')
    if targets.shape != original.shape or len(stamps) != len(original):
        raise ValueError('Original, processed actions and timestamps must have matching lengths/shapes')
    replay_actions.strict_times((stamps-stamps[0])/1e9, len(original), 'original times')
    output = args.output_dir or args.dataset/'replays'/f'{stem}_{time.strftime("%Y%m%d_%H%M%S")}'
    if output.exists():
        raise FileExistsError(f'Replay output already exists: {output}')
    output.mkdir(parents=True)
    comparison = ReplayComparison(original, stamps, args.playback_speed, output, show=not args.no_view,
                                  warn_margins=not args.strict_limits)
    command = [str(actions), '--arm-only', '--playback-speed', str(args.playback_speed),
               '--rate-hz', str(args.rate_hz), '--piper-channel', args.piper_channel, '--output', str(output/'plan.npz')]
    command += ['--max-joint-speed-deg-s', str(args.max_joint_speed_deg_s)]
    if not args.strict_limits:
        command += ['--warn-margins', '--warn-motion-limits']
    if args.preposition:
        command.append('--preposition')
    if args.start_joints_deg is not None:
        command += ['--start-joints-deg', *map(str, args.start_joints_deg)]
    if args.execute:
        command.append('--execute')
    completed = False
    try:
        result = replay_actions.main(command, playback_observer=comparison if args.execute else None)
        completed = result == 0
        return result
    finally:
        comparison.finish(completed)


if __name__ == '__main__':
    raise SystemExit(main())
