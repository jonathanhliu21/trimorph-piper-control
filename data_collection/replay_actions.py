#!/usr/bin/env python3
"""Plan XYZ/rotvec + hand action replay; actuator commands require --execute.

Dataset input: absolute_action_t/NNNNNN.npy [T,6], matching hand and sync files.
Standalone input: [T,19] XYZ/rotvec/hand, with --hz and --pose-frame piper_base.
Saved plan .npz input reuses its absolute joint angles without inverse kinematics.
--relative anchors arm motion at the supplied/measured starting configuration.
Without --start-joints-deg, .npy and relative planning read the current Piper joints.
--preposition prepends a checked joint-space approach to the first arm target.
Planning is not a collision check.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import time
import warnings

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class Limits:
    joint_margin_deg: float = 3.0
    wrist_margin_deg: float = 10.0
    max_joint_step_deg: float = 15.0
    max_joint_speed_deg_s: float = 15.0
    max_position_step_m: float = 0.15
    max_rotation_step_deg: float = 10.0
    min_jacobian_sigma: float = 0.005
    max_hand_step_rad: float = 10000000.0
    max_hand_speed_rad_s: float = 100000.0
    warn_margins: bool = False
    warn_motion_limits: bool = False


class AdvisoryLimitWarning(UserWarning):
    """A configured proximity margin was entered, without exceeding physical bounds."""


def motion_limit(message, limits, error_type=ValueError):
    """Make configured motion thresholds advisory for explicitly selected arm-only replay."""
    if limits.warn_motion_limits:
        warnings.warn(message, AdvisoryLimitWarning)
    else:
        raise error_type(message)


def finite_array(value, columns, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] != columns or not len(value) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite, nonempty [T,{columns}]")
    return value


def strict_times(value, count, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (count,) or not np.isfinite(value).all() or np.any(np.diff(value) <= 0):
        raise ValueError(f"{name} must have {count} strictly increasing finite timestamps")
    return value


def load_actions(path, *, hand_path=None, sync_path=None, hz=None, pose_frame=None, relative=False, arm_only=False):
    """Return times (seconds), arm poses, and calibrated motor 0..12 positions."""
    path = Path(path).expanduser().resolve()
    actions = np.load(path, allow_pickle=False)
    if actions.ndim != 2 or actions.shape[1] not in (6, 19):
        raise ValueError("Action file must be [T,6] or [T,19], with XYZ/rotation-vector first")
    finite_array(actions[:, :6] if arm_only else actions, 6 if arm_only else actions.shape[1], 'actions')
    dataset = path.parent.name == 'absolute_action_t'
    allowed_frames = ('piper_base', 'relative_to_episode_start') if relative else ('piper_base',)
    if sync_path is None and dataset:
        sync_path = path.parent.parent / 'syncs' / (path.stem + '.npz')
    if sync_path is not None:
        with np.load(sync_path, allow_pickle=False) as sync:
            frame = str(sync['pose_frame'])
            if frame not in allowed_frames or (pose_frame is not None and pose_frame != frame):
                raise ValueError("Replay requires pose_frame=piper_base; relative poses must be reprocessed")
            pose_ns = strict_times(sync['pose_monotonic_ns'], len(actions), 'pose timestamps')
            times = (pose_ns - pose_ns[0]) / 1e9
            hand_ns = None if arm_only else np.array(sync['encoder_monotonic_ns'], dtype=float)
    else:
        if pose_frame not in allowed_frames or hz is None or not np.isfinite(hz) or hz <= 0:
            raise ValueError("Standalone actions require --hz and --pose-frame piper_base "
                             "(or relative_to_episode_start with --relative)")
        times = np.arange(len(actions), dtype=float) / hz
        hand_ns = None
    if arm_only:
        return times, actions[:, :6].astype(float), np.zeros((len(actions),13))
    if actions.shape[1] == 19:
        if hand_path is not None:
            raise ValueError('Combined [T,19] actions already contain hand positions')
        hand = actions[:, 6:]
    else:
        if hand_path is None:
            if not dataset:
                raise ValueError('[T,6] actions require --hand-actions and matching timing')
            hand_path = path.parent.parent / 'processed_encoder_t' / path.name
        hand = finite_array(np.load(hand_path, allow_pickle=False), 13, 'hand positions')
        if hand_ns is not None:
            strict_times(hand_ns, len(hand), 'hand timestamps')
            covered = (pose_ns >= hand_ns[0]) & (pose_ns <= hand_ns[-1])
            if not np.any(covered):
                raise ValueError('No overlapping hand/arm timestamps; refusing extrapolation')
            if not np.all(covered):
                print(f'Trimming {int((~covered).sum())} arm rows outside hand timestamp coverage')
                actions, pose_ns = actions[covered], pose_ns[covered]
                times = (pose_ns - pose_ns[0]) / 1e9
            hand = np.column_stack([np.interp(pose_ns - pose_ns[0], hand_ns - pose_ns[0], hand[:, j])
                                    for j in range(13)])
        elif len(hand) != len(actions):
            raise ValueError('Without sync timestamps, hand and arm row counts must match')
    return times, actions[:, :6].astype(float), hand.astype(float)


def anchor_poses(poses, start_pose):
    """Same spatial-delta convention as replay_episode.anchor_relative_piper_stream.

    Start pose uses XYZ/Euler, input/output poses use XYZ/rotation vectors.
    Hand positions remain absolute calibrated motor positions.
    """
    rotations = Rotation.from_rotvec(poses[:, 3:])
    anchored = rotations * rotations[0].inv() * Rotation.from_euler('xyz', start_pose[3:])
    return np.c_[start_pose[:3] + poses[:, :3] - poses[0, :3], anchored.as_rotvec()]


def load_joint_plan(path, *, arm_only=False):
    """Read our replay-plan format, never mistake sync.start_pose for robot joints."""
    with np.load(path, allow_pickle=False) as saved:
        required = {'joints', 'times'} if arm_only else {'joints', 'hand', 'times'}
        if not required.issubset(saved.files):
            raise ValueError('Not a joint replay plan: .npz requires joints [T,6], hand [T,13], times [T]. '
                             'Dataset sync start_pose is XYZ/rotvec, not six joint angles.')
        joints = finite_array(saved['joints'], 6, 'saved joints')
        hand = np.zeros((len(joints),13)) if arm_only else finite_array(saved['hand'], 13, 'saved hand positions')
        times = strict_times(saved['times'], len(joints), 'saved plan times')
        if len(hand) != len(joints):
            raise ValueError('Saved hand and joint row counts must match')
        return times-times[0], joints, hand


def load_hand_actions(path, *, hand_path=None, sync_path=None, hz=None):
    """Hand-only source loading never requires arm poses, joint angles, or Piper."""
    path = Path(path).expanduser().resolve()
    if path.suffix == '.npz':
        with np.load(path,allow_pickle=False) as saved:
            hand = finite_array(saved['hand'],13,'saved hand positions')
            times = strict_times(saved['times'],len(hand),'saved hand times')
        return times-times[0], hand, True
    dataset = path.parent.name in ('absolute_action_t','processed_encoder_t')
    if dataset:
        root = path.parent.parent
        hand_path = hand_path or root/'processed_encoder_t'/path.name
        sync_path = sync_path or root/'syncs'/(path.stem+'.npz')
    source = np.load(hand_path or path,allow_pickle=False)
    if source.ndim != 2 or source.shape[1] not in (13,19):
        raise ValueError('Hand-only replay needs [T,13] hand positions or [T,19] combined actions')
    hand = finite_array(source[:,6:] if source.shape[1] == 19 else source,13,'hand positions')
    if sync_path is not None:
        with np.load(sync_path,allow_pickle=False) as sync:
            stamps = strict_times(sync['encoder_monotonic_ns'],len(hand),'hand timestamps')
        times = (stamps-stamps[0])/1e9
    else:
        if hz is None or not np.isfinite(hz) or hz <= 0:
            raise ValueError('Standalone hand positions require --hz')
        times = np.arange(len(hand))/hz
    return times,hand,False


def plan_hand(times, hand, *, rate_hz=30, playback_speed=.2, limits=Limits()):
    hand = finite_array(hand,13,'hand positions')
    times = strict_times(times,len(hand),'hand timestamps')
    if not np.isfinite(rate_hz) or rate_hz <= 0 or not 0 < playback_speed <= 1:
        raise ValueError('Require positive replay rate and 0 < playback speed <= 1')
    if np.any(np.abs(np.diff(hand,axis=0)) > limits.max_hand_step_rad):
        raise ValueError('Source contains excessive hand position jump')
    duration = (times[-1]-times[0])/playback_speed
    wall = np.arange(int(np.floor(duration*rate_hz))+1)/rate_hz
    if duration > wall[-1]+1e-9:
        wall = np.r_[wall,duration]
    source = np.clip(times[0]+wall*playback_speed,times[0],times[-1])
    sampled = np.column_stack([np.interp(source,times,hand[:,j]) for j in range(13)])
    if len(wall)>1 and np.any(np.abs(np.diff(sampled,axis=0))/np.diff(wall)[:,None] > limits.max_hand_speed_rad_s):
        raise ValueError('Hand replay speed exceeds limit')
    return dict(times=wall,hand=sampled,joints=np.empty((len(wall),0)))


def validate_joint_plan(times, joints, hand, seed, bounds, fk, *, relative=False,
                        playback_speed=1., limits=Limits(), skip_validation=False):
    """Revalidate saved joint targets; absolute is the default, no IK is run."""
    joints = finite_array(joints, 6, 'saved joints').copy()
    hand = finite_array(hand, 13, 'saved hand positions')
    times = strict_times(times, len(joints), 'saved plan times')
    if len(hand) != len(joints) or not 0 < playback_speed <= 1:
        raise ValueError('Require matching rows and 0 < playback speed <= 1')
    seed = np.asarray(seed, dtype=float)
    check_joints(seed, bounds, fk, limits, check_jacobian=not skip_validation)
    if relative:
        joints += seed-joints[0]
    times = (times-times[0])/playback_speed
    steps = np.abs(np.diff(np.vstack([seed, joints]), axis=0))
    if np.any(steps > np.radians(limits.max_joint_step_deg)):
        i,j = np.argwhere(steps > np.radians(limits.max_joint_step_deg))[0]
        source = 'measured/seed start' if i == 0 else f'saved frame {i-1}'
        before = seed[j] if i == 0 else joints[i-1,j]
        motion_limit(f'Saved plan joint jump exceeds limit, including initial target: '
                         f'J{j+1}, {source} -> saved frame {i}, '
                         f'{np.degrees(before):.3f} -> {np.degrees(joints[i,j]):.3f} deg '
                         f'(jump {np.degrees(steps[i,j]):.3f} deg > {limits.max_joint_step_deg:g} deg)', limits)
    if len(times) > 1:
        if np.any(steps[1:]/np.diff(times)[:,None] > np.radians(limits.max_joint_speed_deg_s)):
            motion_limit('Saved plan joint speed exceeds limit', limits)
        if np.any(np.abs(np.diff(hand, axis=0)) > limits.max_hand_step_rad):
            raise ValueError('Saved plan hand position jump exceeds limit')
        if np.any(np.abs(np.diff(hand, axis=0))/np.diff(times)[:,None] > limits.max_hand_speed_rad_s):
            raise ValueError('Saved plan hand speed exceeds limit')
    previous = seed
    poses, sigmas = [], []
    for q in joints:
        sigmas.append(min(check_joints(previous + alpha*(q-previous), bounds, fk, limits,
                                      check_jacobian=not skip_validation)
                          for alpha in np.linspace(0, 1, 5)))
        pose = fk(q)
        poses.append(np.r_[pose[:3], Rotation.from_euler('xyz', pose[3:]).as_rotvec()])
        previous = q
    poses = np.array(poses)
    initial = fk(seed)
    all_poses = np.vstack([np.r_[initial[:3], Rotation.from_euler('xyz', initial[3:]).as_rotvec()], poses])
    if np.any(np.linalg.norm(np.diff(all_poses[:,:3], axis=0), axis=1) > limits.max_position_step_m):
        motion_limit('Saved plan Cartesian position jump exceeds limit', limits)
    rotations = Rotation.from_rotvec(all_poses[:,3:])
    if np.any(np.degrees((rotations[1:]*rotations[:-1].inv()).magnitude()) > limits.max_rotation_step_deg):
        motion_limit('Saved plan Cartesian rotation jump exceeds limit', limits)
    return dict(times=times, joints=joints, hand=hand.copy(), poses=poses,
                jacobian_sigma=np.array(sigmas))


def piper_model():
    # Pure kinematics import: no robot factory or hardware connection.
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    mdh = list(get_mdh('piper'))
    bounds = np.array([ROBOT_JOINT_LIMIT_PRESET_RAD['piper'][f'joint{i}'] for i in range(1, 7)])
    return lambda q: np.asarray(fk_from_mdh(mdh, list(q))), bounds


def jacobian(fk, q):
    """Geometric Jacobian; rotational rows scaled by a 0.2 m characteristic length."""
    base = fk(q)
    rotation = Rotation.from_euler('xyz', base[3:])
    columns = []
    for j in range(6):
        perturbed = q.copy()
        perturbed[j] += 1e-6
        pose = fk(perturbed)
        columns.append(np.r_[(pose[:3] - base[:3]) / 1e-6,
                             .2 * (Rotation.from_euler('xyz', pose[3:]) * rotation.inv()).as_rotvec() / 1e-6])
    return np.column_stack(columns)


def check_joints(q, bounds, fk, limits, *, check_jacobian=True):
    q = np.asarray(q, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('Joint feedback/seed must be six finite radians')
    margin = np.radians(limits.joint_margin_deg)
    lower, upper = bounds[:, 0] + margin, bounds[:, 1] - margin
    invalid = np.flatnonzero((q < lower) | (q > upper))
    if limits.warn_margins and np.any((q < bounds[:, 0]) | (q > bounds[:, 1])):
        raise ValueError('Physical joint limit violated; advisory mode does not permit out-of-bounds joints')
    if len(invalid):
        detail = '; '.join(
            f'J{j+1}={np.degrees(q[j]):.3f} deg, allowed '
            f'[{np.degrees(lower[j]):.3f}, {np.degrees(upper[j]):.3f}] deg '
            f'(bounds [{np.degrees(bounds[j,0]):.3f}, {np.degrees(bounds[j,1]):.3f}], '
            f'margin {limits.joint_margin_deg:g} deg)'
            for j in invalid)
        if limits.warn_margins:
            warnings.warn('Joint margin entered; physical joint bounds remain enforced', AdvisoryLimitWarning)
        else:
            raise ValueError(f'Joint limit or configured joint margin violated: {detail}')
    if abs(q[4]) < np.radians(limits.wrist_margin_deg):
        if limits.warn_margins:
            warnings.warn('Wrist singularity proximity: |J5| below configured margin', AdvisoryLimitWarning)
        else:
            raise ValueError('Wrist singularity margin violated: |J5| too small')
    if not check_jacobian:
        return float('nan')
    sigma = float(np.linalg.svd(jacobian(fk, q), compute_uv=False)[-1])
    if sigma < limits.min_jacobian_sigma:
        if limits.warn_margins:
            warnings.warn('Near-singular Jacobian: minimum singular value below configured threshold', AdvisoryLimitWarning)
        else:
            raise ValueError(f'Near-singular Jacobian: minimum singular value {sigma:.6g}')
    return sigma


def plan_actions(times, poses, hand, seed, bounds, fk, *, rate_hz=30, playback_speed=.2, limits=Limits(), skip_validation=False,
                 preposition=False):
    """Validate source, solve one continuous IK branch, and check joint interpolation."""
    poses = finite_array(poses, 6, 'arm poses')
    hand = finite_array(hand, 13, 'hand positions')
    times = strict_times(times, len(poses), 'action times')
    if len(hand) != len(poses):
        raise ValueError('Hand and arm samples must be aligned')
    if not np.isfinite(rate_hz) or rate_hz <= 0 or not 0 < playback_speed <= 1:
        raise ValueError('Require positive replay rate and 0 < playback speed <= 1')
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (6, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('Joint bounds must be six finite [lower,upper] radian pairs')
    q = np.asarray(seed, dtype=float)
    check_joints(q, bounds, fk, limits, check_jacobian=not skip_validation)
    rotations = Rotation.from_rotvec(poses[:, 3:])
    # Check original rows before resampling so short glitches cannot be skipped.
    distances = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    angles = np.degrees((rotations[1:] * rotations[:-1].inv()).magnitude())
    if np.any(distances > limits.max_position_step_m):
        i = int(np.argmax(distances))
        motion_limit(f'Source contains excessive Cartesian position jump at row {i+1}: {distances[i]*100:.2f} cm', limits)
    if np.any(angles > limits.max_rotation_step_deg):
        i = int(np.argmax(angles))
        motion_limit(f'Source contains excessive Cartesian rotation jump at row {i+1}: {angles[i]:.2f} deg', limits)
    hand_steps = np.abs(np.diff(hand, axis=0))
    if np.any(hand_steps > limits.max_hand_step_rad):
        i, motor = np.unravel_index(hand_steps.argmax(), hand_steps.shape)
        raise ValueError(f'Source contains excessive hand position jump at row {i+1}, motor {motor}: {hand_steps[i,motor]:.4f} rad')
    duration = (times[-1] - times[0]) / playback_speed
    wall = np.arange(int(np.floor(duration * rate_hz)) + 1) / rate_hz
    if duration > wall[-1] + 1e-9:
        wall = np.r_[wall, duration]
    source = np.clip(times[0] + wall * playback_speed, times[0], times[-1])
    xyz = np.column_stack([np.interp(source, times, poses[:, j]) for j in range(3)])
    sampled_hand = np.column_stack([np.interp(source, times, hand[:, j]) for j in range(13)])
    if len(wall) > 1:
        hand_speeds = np.abs(np.diff(sampled_hand, axis=0)) / np.diff(wall)[:, None]
        if np.any(hand_speeds > limits.max_hand_speed_rad_s):
            i, motor = np.unravel_index(hand_speeds.argmax(), hand_speeds.shape)
            raise ValueError(f'Hand replay speed exceeds limit: motor {motor}, '
                             f'{hand_speeds[i,motor]:.4f} rad/s > {limits.max_hand_speed_rad_s:g} rad/s '
                             f'at replay t={wall[i+1]:.3f}s (playback speed {playback_speed:g}). '
                             'Reduce --playback-speed and recheck; joint/pose checks still apply.')
    target_rotations = rotations if len(times) == 1 else Slerp(times, rotations)(source)
    joints, sigmas = [], []
    margin = 0. if limits.warn_margins else np.radians(limits.joint_margin_deg)
    lower, upper = bounds[:, 0] + margin, bounds[:, 1] - margin
    for i, (position, rotation) in enumerate(zip(xyz, target_rotations)):
        def residual(candidate):
            pose = fk(candidate)
            return np.r_[pose[:3] - position,
                         .2 * (Rotation.from_euler('xyz', pose[3:]) * rotation.inv()).as_rotvec()]
        solution = least_squares(residual, q, bounds=(lower, upper), max_nfev=150,
                                 ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = residual(solution.x)
        if not solution.success or np.linalg.norm(error[:3]) > .0005 or np.linalg.norm(error[3:]) / .2 > np.radians(.2):
            detail = ''
            if i == 0:
                start = fk(q)
                gap_mm = np.linalg.norm(start[:3]-position)*1000
                gap_deg = np.degrees((Rotation.from_euler('xyz',start[3:])*rotation.inv()).magnitude())
                detail = f'; starting pose differs from first target by {gap_mm:.1f} mm and {gap_deg:.1f} deg'
            raise ValueError(f'Frame {i}: no accurate IK solution within joint margins '
                             f'(local solve residual {np.linalg.norm(error[:3])*1000:.2f} mm, '
                             f'{np.degrees(np.linalg.norm(error[3:])/.2):.2f} deg){detail}')
        change = np.max(np.abs(solution.x - q))
        allowed = np.radians(limits.max_joint_step_deg)
        if i:
            allowed = min(allowed, np.radians(limits.max_joint_speed_deg_s) * (wall[i] - wall[i-1]))
        if change > allowed and not (i == 0 and preposition):
            motion_limit('Planned joint jump/speed exceeds configured threshold', limits)
        # Test the commanded joint-space segment, including J5 zero crossings.
        sigma = min(check_joints(q + alpha * (solution.x - q), bounds, fk, limits,
                                check_jacobian=not skip_validation)
                    for alpha in np.linspace(0, 1, 5))
        q = solution.x
        joints.append(q.copy())
        sigmas.append(sigma)
    return dict(times=wall, joints=np.array(joints), hand=sampled_hand,
                poses=np.c_[xyz, target_rotations.as_rotvec()], jacobian_sigma=np.array(sigmas))


def prepend_preposition(plan, seed, bounds, fk, limits, *, rate_hz=30, skip_validation=False):
    """Prepend a validated approach at at most 5 deg/s, holding the initial hand target."""
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError('Preposition rate must be finite and positive')
    change = np.max(np.abs(plan['joints'][0] - seed))
    if change < 1e-10:
        return plan
    duration = change / np.radians(min(5., limits.max_joint_speed_deg_s))
    intervals = max(1, int(np.ceil(duration * rate_hz)),
                    int(np.ceil(change / np.radians(limits.max_joint_step_deg / 2))))
    approach_times = np.linspace(0., duration, intervals + 1)
    approach_joints = np.asarray(seed) + np.linspace(0., 1., intervals + 1)[:, None] * (
        plan['joints'][0] - seed)
    approach = validate_joint_plan(
        approach_times, approach_joints, np.repeat(plan['hand'][:1], intervals + 1, axis=0),
        seed, bounds, fk, limits=limits, skip_validation=skip_validation)
    print(f'Preposition: {duration:.2f}s joint-space approach, maximum joint change '
          f'{np.degrees(change):.3f} deg; first target (deg): '
          f'{np.degrees(plan["joints"][0]).round(3).tolist()}', flush=True)
    return {key: np.concatenate((approach[key][:-1],
                                value + duration if key == 'times' else value))
            for key, value in plan.items()}


def read_controller_joint_limits(robot, joint, *, attempts=3):
    """Retry read-only limit queries and preserve the actual failing SDK values."""
    detail = 'no response'
    for attempt in range(1, attempts+1):
        feedback = robot.get_joint_angle_vel_limits(joint, timeout=1, min_interval=0)
        if feedback is None:
            detail = 'no response'
        else:
            raw_low = getattr(feedback.msg, 'min_angle_limit', None)
            raw_high = getattr(feedback.msg, 'max_angle_limit', None)
            detail = f'min_angle_limit={raw_low!r}, max_angle_limit={raw_high!r} (SDK units: radians)'
            try:
                low, high = float(raw_low), float(raw_high)
                if np.isfinite([low, high]).all() and low < high:
                    print(f'Controller J{joint} bounds (deg): [{np.degrees(low):.3f}, {np.degrees(high):.3f}]', flush=True)
                    return low, high
            except (TypeError, ValueError):
                pass
        print(f'WARNING: controller J{joint} limit query {attempt}/{attempts}: {detail}',
              file=sys.stderr, flush=True)
    raise RuntimeError(f'Cannot read valid controller bounds for J{joint} after {attempts} queries: {detail}. '
                       'This is a controller/SDK limit-response problem, not a trajectory limit violation. '
                       'No actuators were enabled.')


def feedback_details(state):
    """Describe the SDK cache without requesting or changing controller state."""
    now = time.time()
    if state is None:
        return 'reason=missing feedback'
    stamp = float(state.timestamp)
    age = now - stamp
    reason = ('nonfinite timestamp' if not np.isfinite(stamp) else
              'future timestamp' if age < 0 else
              'stale feedback' if age > .25 else 'fresh')
    return (f'reason={reason}, now={now:.6f}, timestamp={stamp:.6f}, '
            f'age_ms={age*1000:.3f}, sdk_hz={getattr(state, "hz", "unknown")}')


def read_fresh_joints(api, *, check_freshness=True):
    state = api.get_joint_state()
    if state is None or not np.isfinite(state.timestamp) or (check_freshness and not 0 <= time.time() - state.timestamp <= .25):
        raise RuntimeError('Piper joint feedback missing or older than 250 ms: '
                           + feedback_details(state))
    q = np.asarray(state.joints, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise RuntimeError(f'Invalid Piper joint feedback: joints={q!r}; {feedback_details(state)}')
    return q


def check_status(api, *, check_freshness=True):
    state = api.robot.get_arm_status()
    if state is None or not np.isfinite(state.timestamp) or (check_freshness and not 0 <= time.time() - state.timestamp <= .25):
        raise RuntimeError('Piper status feedback missing or stale: ' + feedback_details(state))
    if int(state.msg.err_code) or any(x in str(state.msg.arm_status) for x in
                                    ('EXCEEDS_LIMIT', 'NO_SOLUTION', 'SINGULARITY', 'COLLISION', 'EMERGENCY')):
        raise RuntimeError(f'Piper controller fault: {state.msg}')


def wait_for_fresh_feedback(api, *, timeout_s=2., check_freshness=True):
    """Let the receiver catch up after offline work, before any actuator enable.

    Only missing/stale caches are retried. Malformed joints and controller faults
    still fail immediately. Playback keeps its immediate freshness checks.
    """
    deadline = time.monotonic() + timeout_s
    announced = False
    while True:
        try:
            joints = read_fresh_joints(api, check_freshness=check_freshness)
            check_status(api, check_freshness=check_freshness)
            return joints
        except RuntimeError as exc:
            if not check_freshness or not str(exc).startswith((
                    'Piper joint feedback missing or older than 250 ms:',
                    'Piper status feedback missing or stale:')):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f'Fresh feedback did not resume within {timeout_s:g}s after planning; '
                                   f'arm was not enabled. Last observation: {exc}') from exc
            if not announced:
                print(f'Waiting up to {timeout_s:g}s for fresh joint/status feedback after planning...', flush=True)
                announced = True
            time.sleep(min(.01, remaining))


def play_plan(plan, api, hand, bounds, fk, limits, *, check_freshness=True, observer=None):
    """Runtime feedback gates. Always uses smooth joint commands, never firmware EEF IK."""
    started = time.monotonic()
    schedule_slip = 0.
    for i, scheduled in enumerate(plan['times']):
        remaining = started + schedule_slip + scheduled - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        elif i and remaining < -.1:
            if limits.warn_motion_limits:
                warnings.warn('Replay deadline missed by >100 ms; shifting schedule to avoid a catch-up burst', AdvisoryLimitWarning)
                schedule_slip -= remaining
            else:
                raise RuntimeError('Replay deadline missed by >100 ms; refusing catch-up burst')
        if api is not None:
            target = plan['joints'][i]
            measured = read_fresh_joints(api,check_freshness=check_freshness)
            check_joints(measured, bounds, fk, limits)
            check_status(api,check_freshness=check_freshness)
            if np.max(np.abs(target - measured)) > np.radians(limits.max_joint_step_deg):
                motion_limit('Measured joints too far from next target', limits, RuntimeError)
            api.move_joint(target.tolist(), mode='smooth')
            if observer is not None:
                plan['runtime_delay_s'] = schedule_slip
                observer(plan, i, time.monotonic() - started, time.monotonic_ns(), measured)
        if hand is not None:
            hand.set_positions(plan['hand'][i], clip=False)


def prepare_hand(args, fingers, first_target):
    from inference.executor.executor import load_hand_class
    from data_collection.collection.hand_common import load_hand_config, verify_connected_hand
    hand_class = load_hand_class(args.hand_api_repo)
    config = replace(load_hand_config(args.hand_config,args.hand_port,tuple(range(13))),operating_mode=3)
    if np.any(fingers < config.joint_lower_limits_array) or np.any(fingers > config.joint_upper_limits_array):
        raise ValueError('Hand actions exceed configured limits; refusing clipping')
    hand = hand_class(config=config,autoconnect=True)
    try:
        verify_connected_hand(hand,role='action replay')
        current = np.asarray(hand.read_pos(),dtype=float)
        # if current.shape != (13,) or not np.isfinite(current).all() or np.max(np.abs(current-first_target))>.1:
        #     raise RuntimeError('Hand is not within 0.1 rad of planned start; no automatic prepositioning')
        hand.configure(enable_torque=False)
        hand.set_motion_profile(profile_velocity_rad_s=.5,profile_acceleration_rad_s2=1.)
        hand.set_positions(current,clip=False)
        hand.enable_torque()
        return hand
    except BaseException:
        hand.close(disable_torque=False)
        raise


def run_hand_only(args):
    times,fingers,saved_plan = load_hand_actions(args.actions,hand_path=args.hand_actions,
        sync_path=args.sync,hz=args.hz)
    speed = args.playback_speed if args.playback_speed is not None else (1. if saved_plan else .2)
    limits = Limits()
    plan = plan_hand(times,fingers,rate_hz=args.rate_hz,playback_speed=speed,limits=limits)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,**plan,devices='hand',limits_json=json.dumps(vars(limits)))
    args.output.with_suffix('.json').write_text(json.dumps(dict(devices='hand',samples=len(plan['times']),
        duration_s=float(plan['times'][-1]),limits=vars(limits)),indent=2))
    print(f"Validated hand-only plan: {len(plan['times'])} samples; saved {args.output}")
    if not args.execute:
        print('Hand-only planning complete; no hardware accessed.')
        return 0
    hand = None
    try:
        hand = prepare_hand(args,fingers,plan['hand'][0])
        play_plan(plan,None,hand,None,None,limits)
        print('Hand replay complete; hand torque remains active. Piper was not accessed.')
        return 0
    finally:
        if hand is not None:
            hand.close(disable_torque=False)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('actions', type=Path)
    p.add_argument('--hand-actions', type=Path)
    p.add_argument('--sync', type=Path)
    p.add_argument('--pose-frame', choices=['piper_base', 'relative_to_episode_start'])
    p.add_argument('--hz', type=float, help='Source rate, only for standalone arrays without sync timestamps')
    p.add_argument('--start-joints-deg', type=float, nargs=6,
                   help='Explicit offline start; otherwise .npy/relative planning reads Piper joints. Execution always reads Piper.')
    p.add_argument('--relative', action='store_true',
                   help='Re-anchor arm motion at starting joints; hand targets stay absolute. Default: absolute targets.')
    p.add_argument('--preposition', action='store_true',
                   help='Prepend a validated joint-space move to the first arm target at at most 5 deg/s; motion requires --execute')
    p.add_argument('--skip-validation',action='store_true',
                   help='Skip costly preflight Jacobian scans. IK, limits, jumps, speeds, wrist margin, and runtime feedback checks remain active.')
    p.add_argument('--debug-feedback', action='store_true',
                   help='Log stage timing and SDK joint/status cache snapshots; does not relax checks or retry failures')
    p.add_argument('--skip-feedback-freshness',action='store_true',
                   help='Temporary override: ignore joint/status timestamp age at startup and during playback; other feedback checks remain active')
    devices = p.add_mutually_exclusive_group()
    devices.add_argument('--arm-only',action='store_true',help='Plan/replay Piper only; do not load or connect to the hand')
    devices.add_argument('--hand-only',action='store_true',help='Plan/replay hand only; do not access Piper or solve IK')
    p.add_argument('--joint-limits-json', type=Path, help='Six [lower,upper] pairs in degrees; execution intersects controller limits')
    p.add_argument('--output', type=Path, default=Path('/tmp/trimorph-action-plan.npz'))
    p.add_argument('--rate-hz', type=float, default=30)
    p.add_argument('--playback-speed', type=float,
                   help='Default: 0.2 for .npy source; 1 for a saved .npz plan (already timed)')
    p.add_argument('--max-joint-speed-deg-s', type=float, default=Limits().max_joint_speed_deg_s,
                   help='Maximum planned speed per arm joint in degrees/second (default: 15)')
    p.add_argument('--warn-margins', action='store_true',
                   help='Treat joint/wrist/Jacobian proximity margins as warnings; physical bounds and runtime faults remain enforced')
    p.add_argument('--warn-motion-limits', action='store_true',
                   help='Arm-only: warn for configured motion thresholds and use controller joint bounds; includes --warn-margins')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Enable and replay on actual hand and Piper after planning')
    mode.add_argument('--check-start', action='store_true',
                      help='Only read Piper joints/status/controller limits and report startup bounds; never enable or replay')
    p.add_argument('--piper-speed', type=int, default=10)
    p.add_argument('--piper-channel', default='can0')
    p.add_argument('--hand-api-repo', type=Path, default=Path(__file__).resolve().parents[2] / 'trimorph-hand-api')
    p.add_argument('--hand-config', type=Path, default=Path(__file__).resolve().parents[1] / 'config/xl330_test_config.yaml')
    p.add_argument('--hand-port', default='/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO9P0-if00-port0')
    args = p.parse_args(argv)
    if args.warn_motion_limits and not args.arm_only:
        p.error('--warn-motion-limits requires --arm-only')
    if not np.isfinite(args.max_joint_speed_deg_s) or args.max_joint_speed_deg_s <= 0:
        p.error('--max-joint-speed-deg-s must be finite and positive')
    if not 1 <= args.piper_speed <= 100:
        p.error('--piper-speed must be 1..100')
    if args.hand_only and (args.relative or args.check_start):
        p.error('--relative and --check-start apply to the arm and cannot be used with --hand-only')
    if args.hand_only and args.preposition:
        p.error('--preposition applies to the arm and cannot be used with --hand-only')
    if args.output.resolve() == args.actions.resolve():
        args.output = args.output.with_name(args.output.stem + '_revalidated.npz')
    if args.output.suffix != '.npz':
        p.error('--output must end in .npz')
    return args


def main(argv=None, *, playback_observer=None):
    args = parse_args(argv)
    if args.skip_feedback_freshness and not args.hand_only:
        print('WARNING: joint/status feedback timestamp-age checks are disabled; '
              'cached feedback may be stale.',flush=True)
    if args.skip_validation and not args.hand_only:
        print('WARNING: --skip-validation skips preflight Jacobian singularity scans. '
              'Joint limits, wrist margins, jumps, speeds, and runtime feedback checks remain active.',flush=True)
    if args.hand_only:
        return run_hand_only(args)
    joint_input = args.actions.suffix == '.npz'
    if joint_input:
        if args.hand_actions is not None or args.sync is not None or args.hz is not None or args.pose_frame is not None:
            raise ValueError('Saved joint plans already include hand positions and timing; do not supply source options')
        times, saved_joints, fingers = load_joint_plan(args.actions,arm_only=args.arm_only)
    else:
        times, poses, fingers = load_actions(args.actions, hand_path=args.hand_actions,
            sync_path=args.sync, hz=args.hz, pose_frame=args.pose_frame, relative=args.relative,arm_only=args.arm_only)
    fk, bounds = piper_model()
    if args.joint_limits_json:
        bounds = np.radians(np.asarray(json.loads(args.joint_limits_json.read_text()), dtype=float))
    planning_bounds = bounds.copy()
    limits = Limits(max_joint_speed_deg_s=args.max_joint_speed_deg_s,
                    warn_margins=args.warn_margins or args.warn_motion_limits,
                    warn_motion_limits=args.warn_motion_limits)
    read_start = args.execute or args.check_start or (
        args.start_joints_deg is None and (args.relative or args.preposition or not joint_input))
    api = hand = None
    enabled = False
    stage = 'connect'
    debug_wall, debug_mono = time.time(), time.monotonic()

    def debug_snapshot(label):
        if not args.debug_feedback:
            return
        elapsed = time.monotonic() - debug_mono
        clock_delta = (time.time() - debug_wall) - elapsed
        print(f'[replay debug] {label}: elapsed_s={elapsed:.3f}, '
              f'wall_minus_monotonic_ms={clock_delta*1000:.3f}, enabled={enabled}',
              file=sys.stderr, flush=True)
        if api is not None:
            # Diagnostics must not mask the original failure if the SDK read fails.
            try:
                state = api.get_joint_state()
                print(f'[replay debug] joints: {feedback_details(state)}; '
                      f'deg={None if state is None else np.degrees(state.joints).round(3).tolist()}',
                      file=sys.stderr, flush=True)
                status = api.robot.get_arm_status()
                print(f'[replay debug] status: {feedback_details(status)}; '
                      f'msg={None if status is None else status.msg}', file=sys.stderr, flush=True)
            except Exception as exc:
                print(f'[replay debug] snapshot failed: {exc}', file=sys.stderr, flush=True)

    try:
        if read_start:
            from scripts.piper_api.piper_api import PiperAPI
            # None avoids changing installation/payload on observation-only connection.
            api = PiperAPI(firmware='v183', channel=args.piper_channel, installation_pos=None, payload=None)
            if not api.connect():
                raise RuntimeError('Piper unavailable')
            stage = 'controller limit queries'
            controller_bounds = np.empty((6, 2))
            for j in range(6):
                low, high = read_controller_joint_limits(api.robot, j+1)
                controller_bounds[j] = [low, high]
                bounds[j] = ([low, high] if args.warn_motion_limits else
                             [max(bounds[j, 0], low), min(bounds[j, 1], high)])
            stage = 'startup feedback and configuration checks'
            debug_snapshot(stage)
            seed = read_fresh_joints(api,check_freshness=not args.skip_feedback_freshness)
            check_status(api,check_freshness=not args.skip_feedback_freshness)
            applied_margin = 0. if limits.warn_margins else limits.joint_margin_deg
            bounds_label = 'controller bounds; software thresholds advisory' if args.warn_motion_limits else 'planning/controller intersection'
            print(f'Startup joints and limits (degrees; {bounds_label}; applied margin={applied_margin:g} deg):')
            for j in range(6):
                print(f'  J{j+1}: measured={np.degrees(seed[j]):.3f} '
                      f'planning={np.degrees(planning_bounds[j]).round(3).tolist()} '
                      f'controller={np.degrees(controller_bounds[j]).round(3).tolist()} '
                      f'allowed={[round(float(np.degrees(bounds[j,0])+applied_margin),3), round(float(np.degrees(bounds[j,1])-applied_margin),3)]}',
                      flush=True)
            try:
                check_joints(seed, bounds, fk, limits)
            except ValueError as exc:
                raise ValueError(f'Starting configuration refused before enabling actuators: {exc}') from exc
            if args.check_start:
                print('Startup checks passed. Read-only check complete; no actuators enabled or targets sent.')
                return 0
        else:
            seed = (saved_joints[0].copy() if joint_input and args.start_joints_deg is None
                    else np.radians(args.start_joints_deg))
        speed = args.playback_speed if args.playback_speed is not None else (1. if joint_input else .2)
        stage = 'planning'
        debug_snapshot(stage)
        if joint_input:
            trajectory_seed = saved_joints[0] if args.preposition and not args.relative else seed
            plan = validate_joint_plan(times, saved_joints, fingers, trajectory_seed, bounds, fk,
                relative=args.relative, playback_speed=speed, limits=limits,skip_validation=args.skip_validation)
        else:
            if args.relative:
                poses = anchor_poses(poses, fk(seed))
            plan = plan_actions(times, poses, fingers, seed, bounds, fk,
                                rate_hz=args.rate_hz, playback_speed=speed, limits=limits,skip_validation=args.skip_validation,
                                preposition=args.preposition)
        if args.preposition:
            plan = prepend_preposition(plan, seed, bounds, fk, limits, rate_hz=args.rate_hz,
                                       skip_validation=args.skip_validation)
        if args.arm_only:
            # Zero-column field prevents an arm-only plan masquerading as a combined plan.
            plan['hand'] = np.empty((len(plan['times']),0))
        debug_snapshot('planning complete')
        stage = 'saving plan and report'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **plan, joint_bounds_rad=bounds, seed_rad=seed,
                 feedback_freshness_checked=not args.skip_feedback_freshness,
                 limits_json=json.dumps(vars(limits)), controller_limits_verified=read_start,
                 devices='arm' if args.arm_only else 'both',
                 preflight_jacobian_checked=not args.skip_validation,
                 replay_mode='relative' if args.relative else 'absolute')
        degrees = np.degrees(plan['joints'])
        report = dict(input=str(args.actions.resolve()), samples=len(degrees),
            feedback_freshness_checked=not args.skip_feedback_freshness,
            preflight_jacobian_checked=not args.skip_validation,
            devices='arm' if args.arm_only else 'both',
            duration_s=float(plan['times'][-1]), controller_limits_verified=read_start,
            seed_source='piper_feedback' if read_start else 'explicit_angles' if args.start_joints_deg is not None else 'saved_plan',
            mode='relative' if args.relative else 'absolute',
            preposition=args.preposition,
            joint_source='saved absolute joints' if joint_input else 'inverse kinematics',
            limits=vars(limits), joint_bounds_deg=np.degrees(bounds).tolist(),
            seed_deg=np.degrees(seed).tolist(), first_joints_deg=degrees[0].tolist(),
            joint_min_deg=degrees.min(axis=0).tolist(), joint_max_deg=degrees.max(axis=0).tolist(),
            min_joint_margin_deg=float(np.minimum(plan['joints']-bounds[:, 0], bounds[:, 1]-plan['joints']).min()*180/np.pi),
            min_jacobian_sigma=None if args.skip_validation else float(plan['jacobian_sigma'].min()))
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
        print(f"{'Planned' if args.skip_validation else 'Validated'} {len(plan['times'])} samples "
              f"over {plan['times'][-1]:.2f}s; saved {args.output}", flush=True)
        sigma_label = 'not checked' if args.skip_validation else f"{report['min_jacobian_sigma']:.6f}"
        print(f"Minimum joint margin {report['min_joint_margin_deg']:.2f} deg; "
              f"minimum weighted Jacobian singular value {sigma_label}", flush=True)
        print('This plan does not check table/tool collisions or certify physical safety.', flush=True)
        if not args.execute:
            if read_start:
                print('Planning only: read-only Piper observation complete; no actuators enabled or targets sent.')
            else:
                print('Offline only: no hardware accessed. Bounds are unverified presets/user values.')
            return 0
        # Prepare optional displays/loggers before fresh-feedback gates and actuator enable.
        if playback_observer is not None and hasattr(playback_observer, 'prepare'):
            playback_observer.prepare(plan, fk)
        # A long solve must never be followed by an unverified initial command.
        stage = 'post-planning feedback and start alignment'
        debug_snapshot(stage)
        measured = wait_for_fresh_feedback(api,check_freshness=not args.skip_feedback_freshness)
        if np.max(np.abs(measured - plan['joints'][0])) > np.radians(limits.max_joint_step_deg):
            motion_limit('Arm is not at the planned start; '
                         f'measured_minus_target_deg={np.degrees(measured-plan["joints"][0]).round(3).tolist()}',
                         limits, RuntimeError)
        check_status(api,check_freshness=not args.skip_feedback_freshness)
        if not args.arm_only:
            hand = prepare_hand(args,fingers,plan['hand'][0])
        # Set speed before enable; all actual targets use move_j's smooth mode.
        api.set_speed_percent(args.piper_speed)
        stage = 'enabling arm'
        enabled = True
        if not api.enable():
            raise RuntimeError('Piper enable failed')
        stage = 'playback'
        debug_snapshot(stage)
        observer_options = {} if playback_observer is None else {'observer': playback_observer}
        play_plan(plan, api, hand, bounds, fk, limits,check_freshness=not args.skip_feedback_freshness,
                  **observer_options)
        print('Replay complete; arm remains at its last target.')
        return 0
    except BaseException:
        if enabled and api is not None:
            try:
                api.robot.electronic_emergency_stop()
            except Exception as exc:
                print(f'WARNING: controller stop failed: {exc}', file=sys.stderr)
        debug_snapshot(f'failure during {stage} (snapshot taken after failure/stop)')
        raise
    finally:
        try:
            if hand is not None:
                hand.close(disable_torque=False)
        finally:
            if api is not None:
                api.disconnect()


if __name__ == '__main__':
    raise SystemExit(main())
