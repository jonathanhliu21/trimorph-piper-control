#!/usr/bin/env python3
"""Plot calibrated Quest-derived targets versus recorded Piper FK, without replay.

    python scripts/quest_piper/plot.py data_trimorph_quest_piper --episode 0

Reads the existing processed action arrays and their shared timestamp clock.
Creates pose_dimensions.png/.pdf, error_dimensions.png/.pdf, and signed_errors.csv
under processed/episode_NNNNNN (or --output-dir). Does not refit calibration,
re-anchor, shift timestamps, or change any source/replay data.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quest_piper.common import errors, metrics, write_csv

QUEST_COLOR, PIPER_COLOR = '#e58700', '#008dab'


def plot_poses(output, stamps, targets, original, note=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    targets, original = np.asarray(targets, float), np.asarray(original, float)
    stamps = np.asarray(stamps, np.int64)
    if (targets.ndim != 2 or targets.shape[1] != 6 or original.shape != targets.shape
            or len(stamps) != len(targets) or len(stamps) < 2
            or not np.isfinite(targets).all() or not np.isfinite(original).all()
            or np.any(np.diff(stamps) <= 0)):
        raise ValueError('Need matching finite [T,6] XYZ/rotvec arrays and strictly increasing timestamps')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    t = (stamps-stamps[0])/1e9
    rq, rp = Rotation.from_rotvec(targets[:, 3:]), Rotation.from_rotvec(original[:, 3:])
    # Euler angles are for the overlay only; errors use SO(3), not Euler subtraction.
    q_angles = np.degrees(np.unwrap(rq.as_euler('xyz'), axis=0))
    p_angles = np.degrees(np.unwrap(rp.as_euler('xyz'), axis=0))
    q_angles += 360*np.round((p_angles[0]-q_angles[0])/360)
    axis_errors = 1000*(targets[:, :3]-original[:, :3])
    rotation_errors = np.degrees((rq*rp.inv()).as_rotvec())
    norms = errors(targets[:, :3], rq, original[:, :3], rp)
    summary = metrics(norms)

    def save(fig, name):
        for extension in ('png', 'pdf'):
            fig.savefig(output/f'{name}.{extension}', dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for j, (axis_name, angle_name) in enumerate(zip('XYZ', ('Roll', 'Pitch', 'Yaw'))):
        for axis, quest, piper, title, unit in (
                (axes[j,0], 1000*targets[:,j], 1000*original[:,j], f'{axis_name} position', 'mm'),
                (axes[j,1], q_angles[:,j], p_angles[:,j], f'{angle_name} (XYZ Euler, unwrapped)', 'deg')):
            axis.plot(t, quest, color=QUEST_COLOR, label='Quest-derived flange target', linewidth=1.6)
            axis.plot(t, piper, color=PIPER_COLOR, linestyle='--', label='Recorded Piper EEF (joint FK)', linewidth=1.4)
            axis.set_title(title, loc='left'); axis.set_ylabel(unit); axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=8); axes[0,1].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('Time since first synchronized sample (s)')
    fig.suptitle('Quest-derived target vs original recorded Piper EEF — one panel per dimension', fontsize=13)
    fig.text(.5, .015, note or 'Piper base frame; matched postprocessing timestamps; no replay feedback.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    save(fig, 'pose_dimensions')

    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    for j, axis_name in enumerate('XYZ'):
        for axis, values, title, unit in (
                (axes[j,0], axis_errors[:,j], f'{axis_name} position difference: Quest-derived − recorded FK', 'mm'),
                (axes[j,1], rotation_errors[:,j], f'{axis_name} component of spatial rotation-error vector', 'deg')):
            axis.axhline(0, color='gray', linewidth=.8)
            axis.plot(t, values, color=QUEST_COLOR, linewidth=1.3)
            axis.set_title(title, loc='left', fontsize=10); axis.set_ylabel(unit); axis.grid(alpha=.25)
    for j, (title, unit, key) in enumerate((('3D position error norm', 'mm', 'position_mm'),
                                           ('SO(3) angular error', 'deg', 'orientation_deg'))):
        axis = axes[3,j]
        axis.plot(t, norms[:,j], color='#b33b38', linewidth=1.3)
        mean, rms = summary['mean'][key], summary['rmse'][key]
        axis.axhline(mean, color='#555555', linestyle='--', label=f'Mean {mean:.3f} {unit}')
        axis.set_title(f'{title} | RMSE {rms:.3f} {unit}', loc='left', fontsize=10)
        axis.set_ylabel(unit); axis.set_xlabel('Time since first synchronized sample (s)')
        axis.legend(fontsize=8); axis.grid(alpha=.25)
    fig.suptitle('Quest-derived target minus original recorded Piper EEF', fontsize=13)
    fig.text(.5, .015, 'Angular components: rotvec(R_Quest-derived × R_recordedᵀ), expressed in Piper base axes; not Euler differences.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    save(fig, 'error_dimensions')
    write_csv(output/'signed_errors.csv', [dict(monotonic_ns=int(stamp), elapsed_s=float(t[i]),
        **{f'{axis.lower()}_error_mm': float(axis_errors[i,j]) for j, axis in enumerate('XYZ')},
        **{f'rotation_{axis.lower()}_error_deg': float(rotation_errors[i,j]) for j, axis in enumerate('XYZ')},
        position_error_mm=float(norms[i,0]), angular_error_deg=float(norms[i,1])) for i, stamp in enumerate(stamps)])
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dataset', type=Path)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args(argv)
    stem = f'{args.episode:06d}'
    folder = args.dataset/'processed'/f'episode_{stem}'
    metadata = json.loads((folder/'metadata.json').read_text())
    targets = np.load(args.dataset/'absolute_action_t'/f'{stem}.npy', allow_pickle=False)
    original = np.load(args.dataset/'original_action_t'/f'{stem}.npy', allow_pickle=False)
    with np.load(args.dataset/'syncs'/f'{stem}.npz', allow_pickle=False) as sync:
        stamps = sync['pose_monotonic_ns']
    mapping = metadata.get('mapping', 'unknown')
    note = (f'Mapping: {mapping}; saved latency {1000*metadata.get("latency_s",0):+.1f} ms applied; '
            + ('first pose anchored to recorded Piper.' if mapping == 'relative_anchored_once' else 'saved absolute calibration.'))
    output = args.output_dir or folder
    summary = plot_poses(output, stamps, targets, original, note)
    print(f'Saved pose_dimensions and error_dimensions (PNG/PDF), signed_errors.csv to {output}')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
