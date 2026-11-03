#!/usr/bin/env python3
"""Offline XYZ/rotvec viewer and joint diagnostics for transformed Piper actions.

Frames are zero-based rows of the original .npy, never resampled camera frames.
IK estimates follow one seed-dependent branch; they are not measured joints or
an executable replay plan. No robot connection is made.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys
import textwrap

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_collection.replay_actions import Limits, finite_array, load_actions, piper_model, strict_times

DEFAULT_SOURCE = REPO_ROOT / 'data_trimorph_pingpong_pick_place_mocap/absolute_action_t'
EVENT_FIELDS = ('frame', 'time_s', 'kind', 'joint', 'previous_frame',
                'previous_deg', 'angle_deg', 'jump_deg', 'threshold_deg',
                'lower_deg', 'upper_deg', 'detail')


def analyze(times, poses, seed, bounds, fk, *, limits=Limits(), progress=False):
    """Scan every source row, continuing after violations and local IK failures.

    Use replay_actions' residual and accuracy tolerances. Initialize with its
    bounded IK, then track an unconstrained branch so out-of-limit angles remain
    visible instead of clipping them away. Failed fits are NaN, are not used as
    seeds, and do not produce joint limit/jump claims.
    """
    poses = finite_array(poses, 6, 'arm poses')
    times = strict_times(times, len(poses), 'pose times')
    seed = np.asarray(seed, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if seed.shape != (6,) or not np.isfinite(seed).all():
        raise ValueError('Seed must contain six finite joint angles')
    if (bounds.shape != (6, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError('Bounds must contain six finite increasing pairs')
    margin = np.radians(limits.joint_margin_deg)
    lower, upper = bounds[:, 0] + margin, bounds[:, 1] - margin
    if np.any(lower >= upper):
        raise ValueError('Joint margin leaves no allowed joint range')
    rotations = Rotation.from_rotvec(poses[:, 3:])
    joints = np.full((len(poses), 6), np.nan)
    errors = np.zeros((len(poses), 2))
    events = []

    def event(i, kind, **fields):
        events.append(dict(frame=i, time_s=float(times[i]), kind=kind, **fields))

    rotation_steps = np.degrees((rotations[1:] * rotations[:-1].inv()).magnitude())
    position_steps = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    q = seed.copy()
    for i, (target, rotation) in enumerate(zip(poses, rotations)):
        def residual(candidate):
            pose = fk(candidate)
            return np.r_[pose[:3] - target[:3], .2 * (
                Rotation.from_euler('xyz', pose[3:]) * rotation.inv()).as_rotvec()]

        options = dict(max_nfev=150, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if i == 0:
            initial = least_squares(residual, np.clip(q, lower, upper),
                                    bounds=(lower, upper), **options)
            candidate = initial.x
        else:
            candidate = q
        solution = least_squares(residual, candidate, **options)
        # Equivalent 2pi representations should not create artificial jumps.
        candidate = q + (solution.x - q + np.pi) % (2 * np.pi) - np.pi
        error = residual(candidate)
        pos_mm = np.linalg.norm(error[:3]) * 1000
        rot_deg = np.degrees(np.linalg.norm(error[3:]) / .2)
        errors[i] = [pos_mm, rot_deg]
        if not solution.success or pos_mm > .5 or rot_deg > .2:
            event(i, 'ik_failure', detail=f'Local IK residual {pos_mm:.3f} mm, '
                  f'{rot_deg:.3f} deg; joint angles unavailable')
        else:
            q = candidate
            joints[i] = q
            for j in np.flatnonzero((q < lower) | (q > upper)):
                hard = q[j] < bounds[j, 0] or q[j] > bounds[j, 1]
                event(i, 'joint_limit' if hard else 'joint_margin', joint=f'J{j+1}',
                      angle_deg=float(np.degrees(q[j])),
                      lower_deg=float(np.degrees(lower[j])),
                      upper_deg=float(np.degrees(upper[j])),
                      detail=f'IK branch; hardware bounds '
                      f'[{np.degrees(bounds[j,0]):.3f}, {np.degrees(bounds[j,1]):.3f}] deg')
            if abs(np.degrees(q[4])) < limits.wrist_margin_deg:
                event(i, 'wrist_margin', joint='J5', angle_deg=float(np.degrees(q[4])),
                      threshold_deg=limits.wrist_margin_deg, detail='|J5| below wrist margin')
            # Only adjacent source rows; a gap is not a one-frame jump.
            if i and np.isfinite(joints[i-1]).all():
                steps = np.abs(np.degrees(q - joints[i-1]))
                for j in np.flatnonzero(steps > limits.max_joint_step_deg):
                    event(i, 'joint_jump', joint=f'J{j+1}', previous_frame=i-1,
                          previous_deg=float(np.degrees(joints[i-1, j])),
                          angle_deg=float(np.degrees(q[j])), jump_deg=float(steps[j]),
                          threshold_deg=limits.max_joint_step_deg,
                          detail='Adjacent source-row IK angle change')
        if i:
            if rotation_steps[i-1] > limits.max_rotation_step_deg:
                event(i, 'rotation_jump', previous_frame=i-1,
                      jump_deg=float(rotation_steps[i-1]),
                      threshold_deg=limits.max_rotation_step_deg,
                      detail='End-effector relative rotation, not a joint angle')
            if position_steps[i-1] > limits.max_position_step_m:
                event(i, 'position_jump', previous_frame=i-1,
                      detail=f'{position_steps[i-1]:.6f} m > {limits.max_position_step_m:g} m')
        if progress and (i % 250 == 0 or i == len(poses)-1):
            print(f'  Analyzed {i+1}/{len(poses)} source frames', flush=True)
    return dict(times=times, poses=poses, joints=joints, errors=errors, events=events)


def describe(event):
    prefix = f"frame {event['frame']} (t={event['time_s']:.6f}s) {event['kind']}"
    if event['kind'] == 'joint_jump':
        return (f"{prefix}: {event['joint']}, frame {event['previous_frame']} -> {event['frame']}, "
                f"{event['previous_deg']:.3f} -> {event['angle_deg']:.3f} deg; "
                f"jump {event['jump_deg']:.3f} > {event['threshold_deg']:g} deg")
    if event['kind'] in ('joint_limit', 'joint_margin'):
        return (f"{prefix}: {event['joint']}={event['angle_deg']:.3f} deg, allowed "
                f"[{event['lower_deg']:.3f}, {event['upper_deg']:.3f}] deg; {event['detail']}")
    if event['kind'] == 'rotation_jump':
        return f"{prefix}: {event['jump_deg']:.3f} > {event['threshold_deg']:g} deg"
    if event['kind'] == 'wrist_margin':
        return f"{prefix}: J5={event['angle_deg']:.3f} deg, |J5| < {event['threshold_deg']:g} deg"
    return f"{prefix}: {event['detail']}"


def visualize(result, bounds, limits, title, output, *, show=True):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    poses, times = result['poses'], result['times']
    angles = np.degrees(result['joints'])
    frames = np.arange(len(poses))
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(4, 3, left=.06, right=.97, top=.90, bottom=.19, hspace=.65, wspace=.4)
    spatial = fig.add_subplot(grid[:3, 0], projection='3d')
    spatial.plot(*poses[:, :3].T, color='0.65', linewidth=1)
    spatial.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Piper-base target path')
    center = (poses[:, :3].max(axis=0) + poses[:, :3].min(axis=0)) / 2
    span = max(float(np.ptp(poses[:, :3], axis=0).max()), .1) * .6
    for setter, c in zip((spatial.set_xlim, spatial.set_ylim, spatial.set_zlim), center):
        setter(c-span, c+span)
    spatial.set_box_aspect((1, 1, 1))
    xyz = fig.add_subplot(grid[3, 0])
    for j, label in enumerate('XYZ'):
        xyz.plot(frames, poses[:, j], label=label)
    xyz.set(xlabel='Source frame (0-based)', ylabel='Position (m)')
    xyz.legend(fontsize=8, ncol=3)
    cursors = [xyz.axvline(0, color='k', alpha=.5)]
    axes = []
    for j in range(6):
        ax = fig.add_subplot(grid[j//2, 1+j%2])
        axes.append(ax)
        ax.plot(frames, angles[:, j], linewidth=1)
        for bound in np.degrees(bounds[j]):
            ax.axhline(bound, color='red', linestyle='--', linewidth=.8)
        for bound in (np.degrees(bounds[j, 0])+limits.joint_margin_deg,
                      np.degrees(bounds[j, 1])-limits.joint_margin_deg):
            ax.axhline(bound, color='orange', linestyle=':', linewidth=.8)
        ax.set(title=f'J{j+1} (estimated deg)', xlabel='Source frame')
        cursors.append(ax.axvline(0, color='k', alpha=.5))
    bad_frames = sorted({e['frame'] for e in result['events']})
    if bad_frames:
        spatial.scatter(*poses[bad_frames, :3].T, c='red', s=8)
    for j, ax in enumerate(axes):
        for kind, color, marker in [('joint_limit', 'red', 'x'), ('joint_margin', 'orange', 'x'),
                                     ('joint_jump', 'magenta', 'o')]:
            idx = [e['frame'] for e in result['events'] if e['kind'] == kind and e.get('joint') == f'J{j+1}']
            ax.scatter(idx, angles[idx, j], color=color, marker=marker, s=15)
    info = fig.text(.36, .30, '', fontsize=9, va='top', family='monospace')
    fig.suptitle(f'{title} — offline, seed-dependent IK\n'
                 'Red: joint limit   Orange: margin   Magenta: joint jump   Dashed: SDK bounds')
    point, = spatial.plot([], [], [], 'ko', markersize=6)
    triad = [spatial.plot([], [], [], color=c, linewidth=2)[0] for c in ('r', 'g', 'b')]
    by_frame = {}
    for e in result['events']:
        by_frame.setdefault(e['frame'], []).append(e)

    def update(value):
        i = int(value)
        position = poses[i, :3]
        point.set_data_3d(*position[:, None])
        rotation = Rotation.from_rotvec(poses[i, 3:]).as_matrix()
        for j, line in enumerate(triad):
            line.set_data_3d(*np.stack([position, position+.04*rotation[:, j]], axis=1))
        for cursor in cursors:
            cursor.set_xdata([i, i])
        details = by_frame.get(i, [])
        info.set_text(f'Frame {i} / {len(poses)-1}   t={times[i]:.4f}s\n'
                      f'XYZ (m): {np.array2string(position, precision=4)}\n'
                      + ('\n'.join(textwrap.shorten(describe(e), width=100, placeholder='…')
                                   for e in details[:3]) if details else 'No flagged events')
                      + (f'\n+{len(details)-3} more events in CSV' if len(details) > 3 else ''))
        fig.canvas.draw_idle()

    slider = Slider(fig.add_axes([.12, .07, .64, .025]), 'Frame', 0, max(1, len(poses)-1), valinit=0, valstep=1)
    slider.on_changed(lambda value: update(min(int(value), len(poses)-1)))
    play = Button(fig.add_axes([.80, .06, .07, .04]), 'Play/Pause')
    next_bad = Button(fig.add_axes([.88, .06, .10, .04]), 'Next issue')
    timer = fig.canvas.new_timer(interval=33)
    playing = [False]

    def advance():
        i = int(slider.val)
        target_time = times[i] + .033
        next_frame = int(np.searchsorted(times, target_time))
        slider.set_val(next_frame if next_frame < len(poses) else 0)

    def toggle(_):
        playing[0] = not playing[0]
        timer.start() if playing[0] else timer.stop()

    def jump(_):
        if bad_frames:
            slider.set_val(next((i for i in bad_frames if i > slider.val), bad_frames[0]))

    timer.add_callback(advance)
    play.on_clicked(toggle)
    next_bad.on_clicked(jump)
    fig.canvas.mpl_connect('close_event', lambda _: timer.stop())
    update(bad_frames[0] if bad_frames else 0)
    slider.set_val(bad_frames[0] if bad_frames else 0)
    fig.savefig(output, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('source', nargs='?', type=Path,
                         help=f'Action .npy or directory (default: {DEFAULT_SOURCE})')
    sources.add_argument('--action-file', '--actions', type=Path,
                         help='Read this action .npy file instead of the default dataset')
    parser.add_argument('--episode', help='Select one episode in a directory, e.g. 3 or 000003')
    parser.add_argument('--start-joints-deg', type=float, nargs=6, default=[0, 45, -45, 0, 30, 0],
                        metavar='DEG', help='Offline IK branch seed; default: 0 45 -45 0 30 0')
    parser.add_argument('--hz', type=float, help='Standalone input rate (dataset uses sync timestamps)')
    parser.add_argument('--pose-frame', choices=['piper_base'], help='Required for standalone files')
    parser.add_argument('--joint-margin-deg', type=float, default=Limits().joint_margin_deg)
    parser.add_argument('--max-joint-step-deg', '--jump-deg', type=float, default=Limits().max_joint_step_deg)
    parser.add_argument('--max-rotation-step-deg', type=float, default=Limits().max_rotation_step_deg)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT/'output/replay_transformed_piper')
    parser.add_argument('--no-show', action='store_true', help='Save plots/reports without opening windows')
    parser.add_argument('--print-limit', type=int, default=30, help='Max events printed per episode; -1 for all')
    args = parser.parse_args(argv)
    if args.action_file is not None:
        if args.episode is not None:
            parser.error('--episode cannot be used with --action-file')
        if args.action_file.suffix != '.npy':
            parser.error('--action-file must specify a .npy file')
    args.source = args.action_file or args.source or DEFAULT_SOURCE
    for name in ('joint_margin_deg', 'max_joint_step_deg', 'max_rotation_step_deg'):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f'--{name.replace("_", "-")} must be finite and nonnegative')
    if args.print_limit < -1:
        parser.error('--print-limit must be -1 or nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    if source.is_dir():
        files = ([source / f'{int(args.episode):06d}.npy'] if args.episode is not None
                 else sorted(source.glob('*.npy')))
    else:
        if args.episode is not None:
            raise ValueError('--episode requires a source directory')
        files = [source]
    if not files:
        raise ValueError(f'No action .npy files in {source}')
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    fk, bounds = piper_model()
    limits = Limits(joint_margin_deg=args.joint_margin_deg,
                    max_joint_step_deg=args.max_joint_step_deg,
                    max_rotation_step_deg=args.max_rotation_step_deg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print('Offline only. Frames = zero-based .npy rows; angles = seed-dependent IK estimates.\n'
          'Out-of-limit IK branches do not prove no other valid branch exists.\n'
          f'IK seed (deg): {args.start_joints_deg}; joint jump threshold: {limits.max_joint_step_deg:g} deg', flush=True)
    for path in files:
        print(f'\n{path.name}', flush=True)
        times, poses, _ = load_actions(path, arm_only=True, hz=args.hz, pose_frame=args.pose_frame)
        result = analyze(times, poses, np.radians(args.start_joints_deg), bounds, fk,
                         limits=limits, progress=True)
        prefix = args.output_dir / path.stem
        with prefix.with_suffix('.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
            writer.writeheader()
            writer.writerows(result['events'])
        np.savez_compressed(prefix.with_suffix('.npz'), times=times, poses=poses,
                            joints=result['joints'], ik_error_mm_deg=result['errors'],
                            bounds_rad=bounds, seed_rad=np.radians(args.start_joints_deg),
                            joint_margin_deg=limits.joint_margin_deg,
                            max_joint_step_deg=limits.max_joint_step_deg,
                            max_rotation_step_deg=limits.max_rotation_step_deg,
                            source=str(path), pose_frame='piper_base')
        print(f"{len(poses)} frames; events: {dict(Counter(e['kind'] for e in result['events']))}")
        events = result['events']
        for event in events if args.print_limit == -1 else events[:args.print_limit]:
            print('  ' + describe(event))
        if args.print_limit != -1 and len(events) > args.print_limit:
            print(f'  {len(events)-args.print_limit} additional events in {prefix.with_suffix(".csv")}')
        print(f'Saving {prefix}.csv / .npz / .png', flush=True)
        visualize(result, bounds, limits, path.stem, prefix.with_suffix('.png'), show=not args.no_show)


if __name__ == '__main__':
    main()
