#!/usr/bin/env python3
"""Convert attached Quest/Piper episodes with a frozen calibration; no hand data.

Writes standard absolute_action_t/*.npy [T,6], syncs/*.npz, processed episode
CSVs, original_action_t/*.npy, and calibrated-Quest-versus-recorded-FK errors.
Default is relative mapping anchored ONCE at the first synchronized FK pose.
--absolute uses the saved calibration origins and controller mounting rotation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_collection.postprocess_dataset import controller_rotation, validate_calibration
from scripts.calib.get_coord_transform import transform_chunk, tracked_point_offset
from scripts.quest_piper.common import (episode_path, read_poses, interpolate, errors, metrics,
                                        pose_rows, write_csv, write_json)


def process(root, index, calibration_path, *, latency_s=None, max_gap=.05, absolute=False, overwrite=False):
    raw = episode_path(root, index)
    metadata = json.loads((raw/'metadata.json').read_text())
    if metadata.get('status') != 'complete':
        raise ValueError(f'{raw}: episode is not complete')
    cal = json.loads(calibration_path.read_text())
    validate_calibration(cal)
    if cal.get('tracker', 'quest') != 'quest':
        raise ValueError('Expected a Quest-to-Piper calibration')
    lag = float(cal.get('latency_s', 0.) if latency_s is None else latency_s)
    if not np.isfinite(lag) or not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError('Require finite latency and positive max-gap')
    qs, q, rq = read_poses(raw/'quest.csv', quest=True)
    ps, p, rp = read_poses(raw/'piper.csv')
    aligned = qs-round(lag*1e9)
    ref_p, ref_r, valid = interpolate(ps, p, rp, aligned, max_gap)
    keep = np.flatnonzero(valid)
    if len(keep) < 2:
        raise ValueError(f'{raw}: fewer than two paired poses')
    # End trimming is permitted; missing data inside the replay path is refused.
    if np.any(np.diff(keep) != 1) or np.any(np.diff(aligned[keep]) > max_gap*1e9):
        raise ValueError(f'{raw}: interior tracking gap; split/re-record the episode or explicitly increase --max-gap')
    stamps, q, rq, ref_p, ref_r = aligned[keep], q[keep], rq[keep], ref_p[keep], ref_r[keep]
    if absolute:
        from scripts.teleop.quest_replay import calibration_origin
        c = controller_rotation(cal)
        if c is None:
            raise ValueError('--absolute needs orientation.controller_rotation in the calibration')
        mapped_r = Rotation.from_matrix(cal['orientation']['rotation_matrix']) * rq * c
        origin, _ = calibration_origin(cal)
        lever = tracked_point_offset(cal)
        mapped_p = q @ np.asarray(cal['linear_matrix']).T + origin
        if lever is not None:
            mapped_p += mapped_r.apply(lever)
    else:
        mapped_p, quat = transform_chunk(q, rq.as_quat(), ref_p[0], ref_r[0].as_quat(), cal)
        mapped_r = Rotation.from_quat(quat)
    difference = errors(mapped_p, mapped_r, ref_p, ref_r)
    output = root/'processed'/raw.name
    actions_path = root/'absolute_action_t'/f'{index:06d}.npy'
    original_path = root/'original_action_t'/f'{index:06d}.npy'
    sync_path = root/'syncs'/f'{index:06d}.npz'
    destinations = (output, actions_path, original_path, sync_path)
    if not overwrite and any(path.exists() for path in destinations):
        raise FileExistsError(f'Outputs for episode {index} exist; use --overwrite')
    output.mkdir(parents=True, exist_ok=True)
    for path in (actions_path, original_path, sync_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(actions_path, np.c_[mapped_p, mapped_r.as_rotvec()].astype(np.float32))
    np.save(original_path, np.c_[ref_p, ref_r.as_rotvec()].astype(np.float32))
    np.savez(sync_path, pose_frame='piper_base', pose_monotonic_ns=stamps,
             monotonic_ns=stamps, pose_idx=np.arange(len(stamps)), quest_idx=keep,
             quest_monotonic_ns=qs[keep], original_pose_monotonic_ns=stamps,
             devices='arm', timestamp_offset_s=-lag)
    origin = int(metadata['started_monotonic_ns'])
    write_csv(output/'piper.csv', pose_rows(stamps, mapped_p, mapped_r, origin))
    write_csv(output/'original_piper.csv', pose_rows(stamps, ref_p, ref_r, origin))
    write_csv(output/'errors.csv', [dict(monotonic_ns=int(t), elapsed_s=(int(t)-int(stamps[0]))/1e9,
                                        position_mm=float(e[0]), orientation_deg=float(e[1]))
                                   for t, e in zip(stamps, difference)])
    report = dict(episode_index=index, pose_frame='piper_base', devices=['arm'],
                  calibration=str(calibration_path.resolve()),
                  calibration_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                  latency_s=lag, mapping='absolute' if absolute else 'relative_anchored_once',
                  samples=len(stamps), trimmed_quest_samples=len(qs)-len(stamps),
                  comparison=metrics(difference),
                  note='Relative mapping forces initial error to zero. FK is a model-based reference, not external ground truth.',
                  actions=str(actions_path), original_actions=str(original_path))
    write_json(output/'metadata.json', report)
    plot(output/'comparison.png', stamps, difference, mapped_p, ref_p)
    from scripts.quest_piper.plot import plot_poses
    plot_poses(output, stamps, np.c_[mapped_p, mapped_r.as_rotvec()], np.c_[ref_p, ref_r.as_rotvec()],
               f'Mapping: {report["mapping"]}; latency {lag*1000:+.1f} ms; first pose anchored in relative mode.')
    print(f'{raw.name}: {len(stamps)} poses; RMSE {report["comparison"]["rmse"]}; {output}')
    return report


def plot(path, stamps, difference, mapped, original, label='Quest-derived'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = (stamps-stamps[0])/1e9
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    axes[0].plot(t, difference[:, 0]); axes[0].set_ylabel('Position error (mm)')
    axes[1].plot(t, difference[:, 1]); axes[1].set_ylabel('Angular error (deg)')
    for j, axis_name in enumerate('XYZ'):
        axes[2].plot(t, mapped[:, j], label=f'{label} {axis_name}')
        axes[2].plot(t, original[:, j], '--', label=f'original FK {axis_name}')
    axes[2].set_ylabel('Piper base position (m)'); axes[2].legend(ncol=3)
    for axis in axes:
        axis.set_xlabel('Recording time (s)'); axis.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', type=Path)
    p.add_argument('--episode', type=int, action='append', help='Repeat to select episodes; default all')
    p.add_argument('--calibration-json', type=Path, required=True)
    p.add_argument('--latency-s', type=float, help='Override saved latency (default: apply saved latency)')
    p.add_argument('--max-gap', type=float, default=.05)
    p.add_argument('--absolute', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)
    indices = args.episode if args.episode is not None else sorted(int(path.name.split('_')[-1]) for path in (args.dataset/'raw').glob('episode_*'))
    if not indices:
        raise ValueError('No raw episodes found')
    for index in indices:
        process(args.dataset, index, args.calibration_json, latency_s=args.latency_s,
                max_gap=args.max_gap, absolute=args.absolute, overwrite=args.overwrite)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
