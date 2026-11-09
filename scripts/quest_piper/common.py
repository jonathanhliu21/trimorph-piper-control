"""Timestamp-preserving pose IO and error metrics shared by Quest/Piper tools."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

POSE_FIELDS = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_poses(path, quest=False):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError(f'{path}: need at least two poses')
    stamps = np.array([int(row['monotonic_ns']) for row in rows], dtype=np.int64)
    if np.any(np.diff(stamps) <= 0):
        raise ValueError(f'{path}: timestamps must be strictly increasing')
    fields = ('px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw') if quest else POSE_FIELDS
    values = np.array([[float(row[key]) for key in fields] for row in rows])
    if not np.isfinite(values).all():
        raise ValueError(f'{path}: nonfinite pose')
    if quest and np.any(np.linalg.norm(values[:, 3:], axis=1) < .5):
        raise ValueError(f'{path}: invalid Quest quaternion')
    rotation = Rotation.from_quat(values[:, 3:]) if quest else Rotation.from_euler('xyz', values[:, 3:])
    return stamps, values[:, :3], rotation


def interpolate(stamps, xyz, rotations, queries, max_gap_s=.05):
    """Linear position/SLERP, exact matches accepted; no extrapolation or long-gap bridging."""
    stamps, queries = np.asarray(stamps, np.int64), np.asarray(queries, np.int64)
    right = np.searchsorted(stamps, queries)
    exact = (right < len(stamps)) & (stamps[np.minimum(right, len(stamps)-1)] == queries)
    upper = np.clip(right, 1, len(stamps)-1)
    lower = upper - 1
    gap = stamps[upper] - stamps[lower]
    valid = exact | ((queries >= stamps[0]) & (queries <= stamps[-1]) & (gap <= max_gap_s * 1e9))
    clipped = np.clip(queries, stamps[0], stamps[-1])
    t = (stamps - stamps[0]) / 1e9
    tq = (clipped - stamps[0]) / 1e9
    position = np.column_stack([np.interp(tq, t, xyz[:, j]) for j in range(3)])
    return position, Slerp(t, rotations)(tq), valid


def errors(xyz, rotations, reference_xyz, reference_rotations):
    return np.column_stack((1000 * np.linalg.norm(xyz-reference_xyz, axis=1),
                            np.degrees((rotations * reference_rotations.inv()).magnitude())))


def metrics(values):
    if not len(values):
        return {'samples': 0}
    return dict(samples=len(values), **{name: dict(zip(('position_mm', 'orientation_deg'), map(float, function(values, axis=0))))
                for name, function in (('rmse', lambda a, axis: np.sqrt(np.mean(a*a, axis=axis))),
                                       ('mean', np.mean), ('p95', lambda a, axis: np.percentile(a, 95, axis=axis)),
                                       ('max', np.max))})


def pose_rows(stamps, xyz, rotations, origin=None):
    origin = int(stamps[0]) if origin is None else origin
    rpy = np.unwrap(rotations.as_euler('xyz'), axis=0)
    return [dict(sample_idx=i, monotonic_ns=int(stamp), elapsed_s=(int(stamp)-origin)/1e9,
                 **dict(zip(POSE_FIELDS, map(float, np.r_[p, r]))))
            for i, (stamp, p, r) in enumerate(zip(stamps, xyz, rpy))]


def episode_path(dataset, index):
    return Path(dataset) / 'raw' / f'episode_{index:06d}'
