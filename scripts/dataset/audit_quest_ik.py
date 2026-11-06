#!/usr/bin/env python3
"""Offline bounded Piper IK audit of processed Quest poses; never connects hardware.

All action rows are checked by default. Reports local numerical IK failures, not
proofs of global unreachability. Jacobian scaling and limits match replay_actions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_collection.replay_actions import Limits, jacobian, piper_model


def model():
    """MDH FK with analytic world geometric Jacobian, verified against SDK FK."""
    from pyAgxArm.utiles.mdh_kinematics import get_mdh
    mdh = np.asarray(get_mdh('piper'))
    sdk_fk, bounds = piper_model()

    def evaluate(q):
        t = np.eye(4)
        origins, axes = [], []
        for angle, (d, a, alpha, offset) in zip(q, mdh):
            ca, sa = np.cos(alpha), np.sin(alpha)
            ct, st = np.cos(angle + offset), np.sin(angle + offset)
            origins.append(t[:3, :3] @ np.array([a, -sa*d, ca*d]) + t[:3, 3])
            axes.append(t[:3, :3] @ np.array([0., -sa, ca]))
            t = t @ np.array([[ct, -st, 0, a], [ca*st, ca*ct, -sa, -sa*d],
                              [sa*st, sa*ct, ca, ca*d], [0, 0, 0, 1]])
        axes = np.asarray(axes).T
        j = np.vstack([np.cross(axes.T, t[:3, 3] - origins).T, .2*axes])
        return t[:3, 3], Rotation.from_matrix(t[:3, :3]), j

    for q in (np.radians([0, 45, -45, 0, 30, 0]), np.radians([-30, 100, -80, 20, 40, 30])):
        xyz, r, j = evaluate(q)
        expected = sdk_fk(q)
        assert np.allclose(xyz, expected[:3], atol=1e-10)
        assert (r * Rotation.from_euler('xyz', expected[3:]).inv()).magnitude() < 1e-10
        assert np.allclose(j, jacobian(sdk_fk, q), atol=1e-6)
    return evaluate, bounds


def audit_episode(args):
    path, stride, seed_deg = args
    path = Path(path)
    poses = np.load(path, allow_pickle=False)
    with np.load(path.parent.parent / 'syncs' / (path.stem + '.npz'), allow_pickle=False) as sync:
        stamps = np.array(sync['pose_monotonic_ns'], dtype=np.int64)
        frame = str(sync['pose_frame'])
    if frame != 'piper_base':
        raise ValueError(f'{path}: IK requires piper_base poses, got {frame}')
    evaluate, bounds = model()
    limits = Limits()
    lower, upper = bounds[:, 0] + np.radians(3), bounds[:, 1] - np.radians(3)
    indices = np.unique(np.r_[np.arange(0, len(poses), stride), len(poses)-1]).astype(int)
    rotations = Rotation.from_rotvec(poses[:, 3:])
    q = np.radians(seed_deg)
    previous_q = previous_index = None
    failures, near, wrist, steps, speeds = [], [], [], [], []
    max_mm = max_deg = max_step = max_speed = 0.
    min_sigma, min_wrist, min_margin = float('inf'), float('inf'), float('inf')
    worst_mm = worst_deg = worst_sigma = None
    for index in indices:
        target, rotation = poses[index, :3], rotations[index]
        cache_q = cache_values = None

        def evaluate_residual(candidate):
            nonlocal cache_q, cache_values
            if cache_q is not None and np.array_equal(candidate, cache_q):
                return cache_values
            xyz, r, geometric = evaluate(candidate)
            rv = (r * rotation.inv()).as_rotvec()
            theta = np.linalg.norm(rv)
            x, y, z = rv
            skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            coefficient = (1/12 + theta**2/720 if theta < 1e-4
                           else (1 - theta / (2*np.tan(theta/2))) / theta**2)
            differential = np.eye(3) - .5*skew + coefficient * (skew @ skew)
            cache_q = candidate.copy()
            cache_values = (np.r_[xyz - target, .2*rv],
                            np.vstack([geometric[:3], differential @ geometric[3:]]))
            return cache_values

        result = least_squares(lambda candidate: evaluate_residual(candidate)[0],
                               np.clip(q, lower+1e-9, upper-1e-9),
                               jac=lambda candidate: evaluate_residual(candidate)[1],
                               bounds=(lower, upper), max_nfev=100,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        q = result.x
        error = evaluate_residual(q)[0]
        mm, deg = np.linalg.norm(error[:3])*1000, np.degrees(np.linalg.norm(error[3:])/.2)
        accurate = bool(mm <= .5 and deg <= .2)
        if mm > max_mm:
            max_mm, worst_mm = float(mm), int(index)
        if deg > max_deg:
            max_deg, worst_deg = float(deg), int(index)
        if not accurate:
            failures.append(int(index))
            previous_q = previous_index = None
            continue
        sigma = float(np.linalg.svd(evaluate(q)[2], compute_uv=False)[-1])
        if sigma < min_sigma:
            min_sigma, worst_sigma = sigma, int(index)
        abs_wrist = abs(float(np.degrees(q[4])))
        min_wrist = min(min_wrist, abs_wrist)
        min_margin = min(min_margin, float(np.degrees(np.minimum(q-bounds[:, 0], bounds[:, 1]-q)).min()))
        if sigma < limits.min_jacobian_sigma:
            near.append(int(index))
        if abs_wrist < limits.wrist_margin_deg:
            wrist.append(int(index))
        if previous_q is not None:
            step = float(np.degrees(np.abs(q-previous_q)).max())
            dt = (int(stamps[index])-int(stamps[previous_index]))/1e9
            speed = step/dt if dt > 0 else float('inf')
            max_step, max_speed = max(max_step, step), max(max_speed, speed)
            if step > limits.max_joint_step_deg:
                steps.append(int(index))
            if speed > limits.max_joint_speed_deg_s:
                speeds.append(int(index))
        previous_q, previous_index = q.copy(), int(index)
    def number(value):
        return float(value) if np.isfinite(value) else None
    return dict(episode=path.stem, total_frames=len(poses), checked_frames=len(indices),
                local_ik_failure_count=len(failures), local_ik_failure_indices=failures,
                near_singular_count=len(near), near_singular_indices=near,
                wrist_margin_failure_count=len(wrist), wrist_margin_failure_indices=wrist,
                excessive_joint_step_count=len(steps), excessive_joint_step_indices=steps,
                excessive_joint_speed_count=len(speeds), excessive_joint_speed_indices=speeds,
                max_position_residual_mm=max_mm, worst_position_frame=worst_mm,
                max_orientation_residual_deg=max_deg, worst_orientation_frame=worst_deg,
                min_scaled_jacobian_sigma=number(min_sigma), worst_sigma_frame=worst_sigma,
                min_abs_joint5_deg=number(min_wrist), min_joint_limit_margin_deg=number(min_margin),
                max_joint_step_deg=max_step, max_joint_speed_deg_s=number(max_speed))


def audit_failure_sample(args):
    """Independent SDK FK and numerical Jacobian, with both wrist seed signs."""
    root, episode, index = args
    pose = np.load(Path(root)/'absolute_action_t'/f'{episode}.npy')[index]
    rotation = Rotation.from_rotvec(pose[3:])
    fk, bounds = piper_model()
    lower, upper = bounds[:, 0]+np.radians(3), bounds[:, 1]-np.radians(3)

    def residual(q):
        actual = fk(q)
        return np.r_[actual[:3]-pose[:3], .2*(
            Rotation.from_euler('xyz', actual[3:])*rotation.inv()).as_rotvec()]

    seeds = np.vstack([np.radians([-30, 100, -80, 0, 40, 30]),
                       np.radians([-30, 100, -80, 0, -40, 30]),
                       np.random.default_rng(42).uniform(lower, upper, (6, 6))])
    results = []
    for seed in seeds:
        fit = least_squares(residual, np.clip(seed, lower+1e-9, upper-1e-9),
                            bounds=(lower, upper), max_nfev=150,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = residual(fit.x)
        results.append((float(np.linalg.norm(error)), fit.x,
                        float(np.linalg.norm(error[:3])*1000),
                        float(np.degrees(np.linalg.norm(error[3:])/.2))))
    best = min(results, key=lambda item: item[0])
    accurate = [item for item in results if item[2] <= .5 and item[3] <= .2]
    result = dict(episode=episode, frame=index, starts=len(seeds), accurate_starts=len(accurate),
                  bounded_position_residual_mm=best[2], bounded_orientation_residual_deg=best[3],
                  bounded_joints_deg=np.degrees(best[1]).tolist())
    if not accurate:
        fit = least_squares(residual, best[1], max_nfev=200,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error, q = residual(fit.x), fit.x
        result.update(unbounded_position_residual_mm=float(np.linalg.norm(error[:3])*1000),
                      unbounded_orientation_residual_deg=float(np.degrees(np.linalg.norm(error[3:])/.2)),
                      unbounded_joints_deg=np.degrees(q).tolist(),
                      unbounded_sdk_limit_violations=(np.flatnonzero(
                          (q < bounds[:, 0]) | (q > bounds[:, 1]))+1).tolist(),
                      unbounded_margin_violations=(np.flatnonzero((q < lower) | (q > upper))+1).tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--episodes', nargs='*')
    parser.add_argument('--seed-deg', nargs=6, type=float,
                        default=[-30., 100., -80., 0., 40., 30.])
    parser.add_argument('--multistart-from', type=Path,
                        help='Audit two failed samples per episode from a prior full audit report')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be positive')
    if args.multistart_from:
        prior = json.loads(args.multistart_from.read_text())
        samples = []
        for episode in prior['episodes']:
            failed = episode['local_ik_failure_indices']
            if failed:
                for index in sorted(set([episode['worst_position_frame'], failed[len(failed)//2]])):
                    samples.append((str(args.dataset_root), episode['episode'], index))
        report = dict(method='Independent SDK FK numerical-Jacobian bounded least_squares; eight starts: '
                      '+/-J5 and six deterministic random seeds. Worst residual and middle failed index per '
                      'failing episode. Failed multistart is not a mathematical proof of unreachability.', samples=[])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for sample in pool.map(audit_failure_sample, samples):
                report['samples'].append(sample)
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2)+'\n')
        return
    paths = sorted((args.dataset_root/'absolute_action_t').glob('*.npy'))
    if args.episodes:
        paths = [path for path in paths if path.stem in args.episodes]
    if not paths:
        parser.error('no action files selected')
    start = time.monotonic()
    report = dict(dataset_root=str(args.dataset_root.resolve()), stride=args.stride,
                  initial_seed_deg=args.seed_deg,
                  limitations='Offline local IK, no collision/dynamics check. A failed local fit is not proof of global unreachability. '
                  'Joint steps/speeds measured only between consecutive accurate sampled solutions; failure gaps excluded. '
                  'Thresholds use replay_actions.Limits; sigma uses rotational Jacobian rows scaled by 0.2 m.',
                  thresholds=dict(position_residual_mm=.5, orientation_residual_deg=.2,
                                  joint_margin_deg=3, wrist_margin_deg=10,
                                  min_scaled_jacobian_sigma=.005, max_joint_step_deg=15,
                                  max_joint_speed_deg_s=15), episodes=[])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for episode in pool.map(audit_episode, [(str(path), args.stride, args.seed_deg) for path in paths]):
            report['episodes'].append(episode)
            report['elapsed_s'] = time.monotonic()-start
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2)+'\n')
            print(f"{episode['episode']}: checked {episode['checked_frames']}, IK failures {episode['local_ik_failure_count']}, "
                  f"sigma {episode['min_scaled_jacobian_sigma']}, wrist {episode['wrist_margin_failure_count']}", flush=True)


if __name__ == '__main__':
    main()
