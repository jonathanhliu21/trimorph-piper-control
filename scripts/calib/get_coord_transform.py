#!/usr/bin/env python3
"""Fit Quest -> Piper relative position and orientation mapping from coordinate_record.py CSV.

    python scripts/calib/get_coord_transform.py output/coords1.csv output/coords2.csv --output output/transform.json

Default: a proper rotation, preserving distances. Use --model similarity to fit
uniform teleoperation gain / unit conversion, or --model linear for independent
axis gains and shear. --allow-reflection permits opposite coordinate handedness
for rigid/similarity fits. All models require motion spanning three dimensions.
Use --fit-offset to estimate a rotating controller-to-flange position correction
from pose recordings. The pose helpers apply it using the Piper starting pose.

For column vectors, delta_piper = A @ delta_quest. The JSON's 4x4 displacement
matrix has zero translation. Use piper_anchor + A @ (quest - quest_anchor) to
apply it around any new pair of reference positions. The separately saved
position matrix includes the fitted offset for the recorded coordinate origins.
Orientation is fitted independently using spatial relative rotations:
    Dq = Rq(t) @ Rq(0).T; Dp = B @ Dq @ B.T
Use Rp(t) = Dp @ Rp(0). Euler angle subtraction is never used.
Position and angular mappings may differ; this is not an SE(3) hand-eye fit.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

POSITION_FIELDS = tuple(f"{device}_{axis}" for device in ("quest", "piper") for axis in "xyz")


def _rotation_from_quat(values):
    values = np.asarray(values, dtype=float)
    if values.shape[-1:] != (4,) or not np.isfinite(values).all():
        raise ValueError("Quaternions must contain four finite xyzw components")
    return Rotation.from_quat(values)


def load_positions(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(("elapsed_s", *POSITION_FIELDS)) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("CSV is missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if len(rows) < 4:
        raise ValueError("CSV needs at least four synchronized samples")
    values = np.array([[float(row[key]) for key in ("elapsed_s", *POSITION_FIELDS)] for row in rows])
    if not np.isfinite(values).all():
        raise ValueError("CSV contains nonfinite timestamps or positions")
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("CSV elapsed_s must be strictly increasing")
    return values[:, 0], values[:, 1:4], values[:, 4:7]


def _recording_groups(sample_count, recording_lengths, weighting):
    if weighting not in ("equal", "samples"):
        raise ValueError("weighting must be equal or samples")
    lengths = [sample_count] if recording_lengths is None else list(recording_lengths)
    if not lengths or any(length < 4 for length in lengths) or sum(lengths) != sample_count:
        raise ValueError("Recording lengths must sum to sample count, with at least four samples each")
    edges = np.cumsum([0, *lengths])
    groups = [slice(int(start), int(end)) for start, end in zip(edges[:-1], edges[1:])]
    weights = np.concatenate([np.full(length, 1. / length if weighting == "equal" else 1.)
                              for length in lengths])
    return groups, weights / weights.sum()


def _position_errors(residual, weights):
    return {
        "position_rmse": float(np.sqrt(np.sum(weights * np.sum(residual**2, axis=1)))),
        "position_axis_rmse": np.sqrt(np.sum(weights[:, None] * residual**2, axis=0)).tolist(),
        "position_max": float(np.max(np.linalg.norm(residual, axis=1))),
    }


def fit_transform(quest, piper, *, model="rigid", allow_reflection=False,
                  recording_lengths=None, weighting="equal"):
    """Least squares fit after centering, so origins cannot affect the mapping."""
    quest, piper = np.asarray(quest, dtype=float), np.asarray(piper, dtype=float)
    if quest.ndim != 2 or quest.shape[1] != 3 or piper.shape != quest.shape or len(quest) < 4:
        raise ValueError("Need matching Nx3 arrays with at least four samples")
    if not np.isfinite(quest).all() or not np.isfinite(piper).all():
        raise ValueError("Positions must be finite")
    if model not in ("rigid", "similarity", "linear"):
        raise ValueError(f"Unknown model: {model}")
    groups, weights = _recording_groups(len(quest), recording_lengths, weighting)
    qmeans = [quest[group].mean(axis=0) for group in groups]
    pmeans = [piper[group].mean(axis=0) for group in groups]
    q = np.concatenate([quest[group] - mean for group, mean in zip(groups, qmeans)])
    p = np.concatenate([piper[group] - mean for group, mean in zip(groups, pmeans)])
    qw, pw = q * np.sqrt(weights[:, None]), p * np.sqrt(weights[:, None])
    spectra = {}
    for name, points in (("quest", qw), ("piper", pw)):
        singular = np.linalg.svd(points, compute_uv=False)
        # Reject nearly planar/linear motion; weak axes make calibration unstable.
        if singular[0] <= 1e-9 or singular[-1] <= max(1e-9, singular[0] * 1e-3):
            raise ValueError(f"{name} motion must span all three axes; record a larger 3D trajectory")
        spectra[name] = singular.tolist()
    scale = None
    if model == "linear":
        a = np.linalg.lstsq(qw, pw, rcond=None)[0].T
    else:
        u, singular, vt = np.linalg.svd(qw.T @ pw)
        signs = np.ones(3)
        if not allow_reflection and np.linalg.det(vt.T @ u.T) < 0:
            signs[-1] = -1
        rotation = vt.T @ np.diag(signs) @ u.T
        scale = float(np.sum(singular * signs) / np.sum(qw * qw)) if model == "similarity" else 1.0
        a = scale * rotation
    offsets = [p0 - a @ q0 for q0, p0 in zip(qmeans, pmeans)]
    offset = offsets[0]
    displacement = np.eye(4)
    displacement[:3, :3] = a
    position = displacement.copy()
    position[:3, 3] = offset
    residual = q @ a.T - p
    step_residuals = [np.diff(quest[group], axis=0) @ a.T - np.diff(piper[group], axis=0)
                      for group in groups]
    step_means = [float(np.mean(np.sum(error**2, axis=1))) for error in step_residuals]
    step_weights = None if weighting == "equal" else [len(error) for error in step_residuals]
    fit_errors = _position_errors(residual, weights)
    fit_errors["consecutive_displacement_rmse"] = float(np.sqrt(np.average(step_means, weights=step_weights)))
    recordings = []
    for group, qmean, pmean, recording_offset, steps in zip(groups, qmeans, pmeans, offsets, step_means):
        errors = _position_errors(residual[group], np.full(group.stop - group.start, 1. / (group.stop - group.start)))
        errors["consecutive_displacement_rmse"] = float(np.sqrt(steps))
        recordings.append(dict(sample_count=group.stop - group.start,
                               quest_reference_position=qmean.tolist(), piper_reference_position=pmean.tolist(),
                               position_offset=recording_offset.tolist(), fit_errors=errors))
    return {
        "direction": "quest_to_piper",
        "model": model,
        "sample_count": len(quest),
        "equation": "delta_piper = linear_matrix @ delta_quest (column vectors)",
        "anchored_equation": "piper = piper_anchor + linear_matrix @ (quest - quest_anchor)",
        "units": "Native CSV units; errors in Piper position units",
        "linear_matrix": a.tolist(),
        "displacement_matrix_4x4": displacement.tolist(),
        "position_matrix_4x4": position.tolist() if len(groups) == 1 else None,
        "quest_reference_position": qmeans[0].tolist() if len(groups) == 1 else None,
        "piper_reference_position": pmeans[0].tolist() if len(groups) == 1 else None,
        "position_offset": offset.tolist() if len(groups) == 1 else None,
        "uniform_scale": scale,
        "determinant": float(np.linalg.det(a)),
        "axis_gains_singular_values": np.linalg.svd(a, compute_uv=False).tolist(),
        "motion_singular_values": spectra,
        "fit_errors": fit_errors,
        "recording_count": len(groups),
        "weighting": weighting,
        "recordings": recordings,
        "limitations": "Training residuals only; no latency correction. Position fit alone does not calibrate orientation. "
                       "Use a separate trajectory to validate. A negative determinant changes handedness.",
    }


def load_poses(path):
    """Load synchronized positions and normalized xyzw quaternions."""
    times, quest, piper = load_positions(path)
    fields = [f"{device}_{component}" for device in ("quest", "piper")
              for component in ("qx", "qy", "qz", "qw")]
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(fields) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("CSV lacks orientation columns; record a new pose CSV or use --position-only: "
                             + ", ".join(sorted(missing)))
        values = np.array([[float(row[field]) for field in fields] for row in reader])
    quest_quats = _rotation_from_quat(values[:, :4]).as_quat()
    piper_quats = _rotation_from_quat(values[:, 4:]).as_quat()
    return times, quest, piper, quest_quats, piper_quats


def fit_orientation_transform(quest_quaternions, piper_quaternions, *,
                              recording_lengths=None, weighting="equal"):
    """Fit proper B in Dp = B Dq B.T, cancelling fixed tool-frame offsets.

    D is a spatial/world relative rotation: R(t) R(0)^T. Rotation vectors
    transform as B @ rotvec(Dq). Unlike position scale/shear, B preserves angles.
    """
    quest = _rotation_from_quat(quest_quaternions).as_matrix()
    piper = _rotation_from_quat(piper_quaternions).as_matrix()
    if quest.ndim != 3 or quest.shape != piper.shape or len(quest) < 4:
        raise ValueError("Need matching Nx4 orientation arrays with at least four samples")
    groups, weights = _recording_groups(len(quest), recording_lengths, weighting)
    dq = np.concatenate([quest[group] @ quest[group.start].T for group in groups])
    dp = np.concatenate([piper[group] @ piper[group.start].T for group in groups])
    qvec = Rotation.from_matrix(dq).as_rotvec()
    pvec = Rotation.from_matrix(dp).as_rotvec()
    # At pi the log's axis sign is ambiguous. Exclude these samples from fitting.
    usable = ((np.linalg.norm(qvec, axis=1) < np.pi - 1e-4)
              & (np.linalg.norm(pvec, axis=1) < np.pi - 1e-4))
    q = qvec[usable] * np.sqrt(weights[usable, None])
    p = pvec[usable] * np.sqrt(weights[usable, None])
    spectra = {}
    for name, vectors in (("quest", q), ("piper", p)):
        singular = np.linalg.svd(vectors, compute_uv=False)
        if len(singular) < 2 or singular[1] <= max(1e-4, singular[0] * 1e-3):
            raise ValueError(f"{name} orientation needs rotations about at least two nonparallel axes")
        spectra[name] = singular.tolist()
    u, _, vt = np.linalg.svd(q.T @ p)
    signs = np.ones(3)
    signs[-1] = np.linalg.det(vt.T @ u.T)
    b = vt.T @ np.diag(signs) @ u.T
    predicted = b @ dq @ b.T
    errors = Rotation.from_matrix(predicted @ np.swapaxes(dp, -1, -2)).magnitude()
    return {
        "rotation_matrix": b.tolist(),
        "equation": "Dp = B @ Dq @ B.T; Dq = Rq(t) @ Rq(start).T; Rp(t) = Dp @ Rp(start)",
        "quaternion_order": "xyzw",
        "rotation_convention": "active rotation matrices; spatial/world relative rotations",
        "angular_scale": 1.0,
        "motion_singular_values_rad": spectra,
        "fit_rmse_rad": float(np.sqrt(np.sum(weights * errors**2))),
        "fit_max_rad": float(np.max(errors)),
        "recordings": [dict(fit_rmse_rad=float(np.sqrt(np.mean(errors[group]**2))),
                            fit_max_rad=float(np.max(errors[group]))) for group in groups],
        "limitations": "Independent angular calibration, angle preserving. "
                       "Angular gain, latency and tracked-point lever arms are not fitted. "
                       "For a reflected positional basis C, the equivalent axial basis is det(C)*C.",
    }


def fit_tracked_point_offset(quest, piper, piper_quaternions, result, *,
                             recording_lengths=None, weighting="equal", allow_reflection=False):
    """Fit p = A q + Rp l + b, with l fixed in Piper flange coordinates.

    Center every recording independently to remove its constant origin b.
    Both rotation parities are optimized when reflections are allowed.
    """
    quest, piper = np.asarray(quest, dtype=float), np.asarray(piper, dtype=float)
    rotations = _rotation_from_quat(piper_quaternions).as_matrix()
    groups, weights = _recording_groups(len(quest), recording_lengths, weighting)
    qmeans = [quest[g].mean(axis=0) for g in groups]
    pmeans = [piper[g].mean(axis=0) for g in groups]
    rmeans = [rotations[g].mean(axis=0) for g in groups]
    q = np.concatenate([quest[g] - mean for g, mean in zip(groups, qmeans)])
    p = np.concatenate([piper[g] - mean for g, mean in zip(groups, pmeans)])
    r = np.concatenate([rotations[g] - mean for g, mean in zip(groups, rmeans)])
    root_weights = np.sqrt(weights[:, None])
    model = result["model"]
    initial_a = np.asarray(result["linear_matrix"])

    if model == "linear":
        def unpack(z):
            return z[:9].reshape(3, 3), z[9:12]
        initials = [np.r_[initial_a.ravel(), np.zeros(3)]]
        bases = [None]
    else:
        initials, bases = [], []
        for parity in ([1., -1.] if allow_reflection else [1.]):
            basis = np.diag([parity, 1., 1.])
            u, _, vt = np.linalg.svd(initial_a @ basis)
            proper = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
            initial = np.r_[Rotation.from_matrix(proper).as_rotvec(), np.zeros(3)]
            if model == "similarity":
                initial = np.r_[initial, np.log(max(abs(result["uniform_scale"]), 1e-6))]
            initials.append(initial)
            bases.append(basis)

    candidates = []
    for initial, basis in zip(initials, bases):
        if model != "linear":
            def unpack(z):
                scale = np.exp(z[6]) if model == "similarity" else 1.
                return scale * Rotation.from_rotvec(z[:3]).as_matrix() @ basis, z[3:6]

        def residual(z):
            a, offset = unpack(z)
            return ((q @ a.T + r @ offset - p) * root_weights).ravel()

        fit = least_squares(residual, initial, max_nfev=1000)
        if fit.success:
            a, offset = unpack(fit.x)
            candidates.append((fit.cost, a, offset, fit.jac))
    if not candidates:
        raise ValueError("Tracked-point offset fit failed to converge")
    _, a, offset, jacobian = min(candidates, key=lambda item: item[0])
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular[-1] <= max(1e-9, singular[0] * 1e-6):
        raise ValueError("Tracked-point offset is underdetermined; record independent 3D translations "
                         "and rotations about multiple nonparallel axes")
    if np.linalg.matrix_rank(a) < 3:
        raise ValueError("Tracked-point fit produced a singular position mapping")

    errors = q @ a.T + r @ offset - p
    result["fit_errors"] = _position_errors(errors, weights)
    step_errors, offsets = [], []
    for g, qm, pm, rm, recording in zip(groups, qmeans, pmeans, rmeans, result["recordings"]):
        steps = np.diff(errors[g], axis=0)
        step_errors.append(float(np.mean(np.sum(steps**2, axis=1))))
        recording["fit_errors"] = _position_errors(errors[g], np.full(g.stop - g.start, 1. / (g.stop - g.start)))
        recording["fit_errors"]["consecutive_displacement_rmse"] = float(np.sqrt(step_errors[-1]))
        offsets.append(pm - a @ qm - rm @ offset)
        recording["position_offset"] = offsets[-1].tolist()
        recording["piper_mean_rotation_matrix"] = rm.tolist()
    step_weights = None if weighting == "equal" else [g.stop - g.start - 1 for g in groups]
    result["fit_errors"]["consecutive_displacement_rmse"] = float(np.sqrt(np.average(step_errors, weights=step_weights)))
    displacement = np.eye(4)
    displacement[:3, :3] = a
    result.update(
        linear_matrix=a.tolist(), displacement_matrix_4x4=displacement.tolist(),
        position_matrix_4x4=None,
        position_offset=offsets[0].tolist() if len(groups) == 1 else None,
        uniform_scale=float(np.linalg.svd(a, compute_uv=False)[0]) if model == "similarity" else (1. if model == "rigid" else None),
        determinant=float(np.linalg.det(a)),
        axis_gains_singular_values=np.linalg.svd(a, compute_uv=False).tolist(),
        equation="piper = A @ quest + R_piper @ tracked_point_offset.vector + origin_offset",
        anchored_equation="piper = piper_start + A @ (quest - quest_start) + (R_piper - R_piper_start) @ tracked_point_offset.vector",
        tracked_point_offset=dict(vector=offset.tolist(), frame="piper_flange",
                                  units="Piper position units (meters with pyAgxArm)",
                                  length=float(np.linalg.norm(offset)),
                                  convention="Add R_piper @ vector to the mapped Quest point to obtain the Piper flange point"),
        offset_fit_jacobian_singular_values=singular.tolist(),
    )
    return result


def tracked_point_offset(calibration):
    """Read and validate the optional rotating position correction."""
    if "tracked_point_offset" not in calibration:
        return None
    info = calibration["tracked_point_offset"]
    if not isinstance(info, dict) or info.get("frame") != "piper_flange":
        raise ValueError("tracked_point_offset.frame must be piper_flange")
    vector = np.asarray(info.get("vector"), dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("tracked_point_offset.vector must contain three finite values")
    return vector


def fit_pose_transform(quest, piper, quest_quaternions, piper_quaternions, *,
                       recording_lengths=None, weighting="equal", fit_offset=False, **kwargs):
    if len(quest_quaternions) != len(quest) or len(piper_quaternions) != len(piper):
        raise ValueError("Position and orientation sample counts must match")
    result = fit_transform(quest, piper, recording_lengths=recording_lengths, weighting=weighting, **kwargs)
    if fit_offset:
        result = fit_tracked_point_offset(quest, piper, piper_quaternions, result,
                                          recording_lengths=recording_lengths, weighting=weighting,
                                          allow_reflection=kwargs.get("allow_reflection", False))
    result["orientation"] = fit_orientation_transform(
        quest_quaternions, piper_quaternions, recording_lengths=recording_lengths, weighting=weighting)
    for recording, angular in zip(result["recordings"], result["orientation"]["recordings"]):
        recording["orientation_errors"] = angular
    result["schema_version"] = 3 if fit_offset else 2
    result["relative_frame"] = "world"
    result["limitations"] = ("Training residuals only; validate on another trajectory. "
                             + ("No latency correction. Tracked-point offset estimated from measured Piper orientations. "
                                if fit_offset else "No latency correction or tracked-point offset calibration. ") +
                             "Position A and orientation B are independent; not a single SE(3) transform.")
    if fit_offset:
        # Conversion uses inferred orientations, so report that error as well as
        # the position fit's measured-orientation residuals.
        groups, weights = _recording_groups(len(quest), recording_lengths, weighting)
        centered_errors, anchored_errors = [], []
        for group, recording in zip(groups, result["recordings"]):
            targets, _ = transform_chunk(np.asarray(quest)[group], np.asarray(quest_quaternions)[group],
                                         np.asarray(piper)[group.start], np.asarray(piper_quaternions)[group.start], result)
            error = targets - np.asarray(piper)[group]
            centered = error - error.mean(axis=0)
            local_weights = np.full(len(error), 1. / len(error))
            recording["fit_errors"]["mapped_orientation_position_rmse"] = _position_errors(centered, local_weights)["position_rmse"]
            recording["fit_errors"]["anchored_pose_position_rmse"] = _position_errors(error, local_weights)["position_rmse"]
            centered_errors.append(centered)
            anchored_errors.append(error)
        result["fit_errors"]["mapped_orientation_position_rmse"] = _position_errors(np.concatenate(centered_errors), weights)["position_rmse"]
        result["fit_errors"]["anchored_pose_position_rmse"] = _position_errors(np.concatenate(anchored_errors), weights)["position_rmse"]
    return result


def map_relative_pose(delta_positions, delta_quaternions, calibration, *, relative_frame="world",
                      quest_start_quaternion=None, piper_start_quaternion=None):
    """Map chunk deltas (one pose or batches), returning position and xyzw deltas.

    world: dp = p(t)-p(0), D = R(t) R(0)^T.
    chunk: dp = R(0)^T (p(t)-p(0)), D = R(0)^T R(t).
    For chunk coordinates supply both starting orientations; output deltas are
    expressed in the Piper chunk-start frame. No recorded position offset is used.
    A tracked-point correction additionally requires the Piper starting
    orientation, including for world coordinates.
    """
    if relative_frame not in ("world", "chunk"):
        raise ValueError("relative_frame must be world or chunk")
    a = np.asarray(calibration["linear_matrix"], dtype=float)
    if "orientation" not in calibration:
        raise ValueError("Calibration lacks an orientation fit; fit a pose CSV first")
    b = np.asarray(calibration["orientation"]["rotation_matrix"], dtype=float)
    offset = tracked_point_offset(calibration)
    positions = np.asarray(delta_positions, dtype=float)
    rotations = _rotation_from_quat(delta_quaternions).as_matrix()
    if positions.shape[-1:] != (3,) or positions.shape[:-1] != rotations.shape[:-2]:
        raise ValueError("Position and quaternion delta shapes must match")
    if not np.isfinite(positions).all():
        raise ValueError("Position deltas must be finite")
    if relative_frame == "chunk":
        if quest_start_quaternion is None or piper_start_quaternion is None:
            raise ValueError("chunk coordinates require both starting orientations")
        rq0 = _rotation_from_quat(quest_start_quaternion).as_matrix()
        rp0 = _rotation_from_quat(piper_start_quaternion).as_matrix()
        if rq0.shape != (3, 3) or rp0.shape != (3, 3):
            raise ValueError("Starting orientations must each be one quaternion")
        positions = positions @ rq0.T
        rotations = rq0 @ rotations @ rq0.T
    mapped_positions = positions @ a.T
    mapped_rotations = b @ rotations @ b.T
    if offset is not None:
        if piper_start_quaternion is None:
            raise ValueError("Tracked-point offset requires piper_start_quaternion")
        rp0 = _rotation_from_quat(piper_start_quaternion).as_matrix()
        if rp0.shape != (3, 3):
            raise ValueError("Piper starting orientation must be one quaternion")
        initial_offset = rp0 @ offset
        mapped_positions = mapped_positions + mapped_rotations @ initial_offset - initial_offset
    if relative_frame == "chunk":
        mapped_positions = mapped_positions @ rp0
        mapped_rotations = rp0.T @ mapped_rotations @ rp0
    return mapped_positions, Rotation.from_matrix(mapped_rotations).as_quat()


def transform_chunk(quest_positions, quest_quaternions, piper_start_position,
                    piper_start_quaternion, calibration):
    """Convert an absolute Quest pose chunk to absolute Piper targets.

    Every target is relative to the FIRST Quest pose and the supplied Piper
    starting pose. Targets never accumulate per-step integration drift.
    """
    positions = np.asarray(quest_positions, dtype=float)
    matrices = _rotation_from_quat(quest_quaternions).as_matrix()
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("Quest positions must be a nonempty Nx3 chunk")
    if matrices.shape != (len(positions), 3, 3):
        raise ValueError("Quest orientations must be matching Nx4 quaternions")
    p0 = np.asarray(piper_start_position, dtype=float)
    rp0 = _rotation_from_quat(piper_start_quaternion).as_matrix()
    if p0.shape != (3,) or not np.isfinite(p0).all() or rp0.shape != (3, 3):
        raise ValueError("Supply one finite Piper starting position and quaternion")
    dpositions, dquats = map_relative_pose(
        positions - positions[0],
        Rotation.from_matrix(matrices @ matrices[0].T).as_quat(), calibration,
        piper_start_quaternion=piper_start_quaternion)
    targets = _rotation_from_quat(dquats).as_matrix() @ rp0
    return p0 + dpositions, Rotation.from_matrix(targets).as_quat()


def fit_csvs(paths, *, position_only=False, model="rigid", allow_reflection=False, weighting="equal", fit_offset=False):
    """Joint fit with independent references and equal recording weight by default."""
    paths = [Path(path).expanduser() for path in paths]
    if position_only and fit_offset:
        raise ValueError("--fit-offset requires orientation data; omit --position-only")
    if not paths:
        raise ValueError("Provide at least one CSV")
    streams = []
    for path in paths:
        try:
            streams.append(load_positions(path) if position_only else load_poses(path))
        except (ValueError, OSError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    lengths = [len(stream[0]) for stream in streams]
    options = dict(model=model, allow_reflection=allow_reflection,
                   recording_lengths=lengths, weighting=weighting)
    quest = np.concatenate([stream[1] for stream in streams])
    piper = np.concatenate([stream[2] for stream in streams])
    if position_only:
        result = fit_transform(quest, piper, **options)
    else:
        result = fit_pose_transform(quest, piper,
                                   np.concatenate([stream[3] for stream in streams]),
                                   np.concatenate([stream[4] for stream in streams]), fit_offset=fit_offset, **options)
    for path, stream, recording in zip(paths, streams, result["recordings"]):
        recording.update(input_csv=str(path), duration_s=float(stream[0][-1] - stream[0][0]))
    result["input_csvs"] = [str(path) for path in paths]
    if len(paths) == 1:
        result["input_csv"] = str(paths[0])
    result["duration_s"] = sum(recording["duration_s"] for recording in result["recordings"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, nargs="+", help="One or more CSVs produced by coordinate_record.py")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON (use -f to overwrite)")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite an existing output file")
    parser.add_argument("--model", choices=("rigid", "similarity", "linear"), default="rigid")
    parser.add_argument("--allow-reflection", action="store_true", help="Allow opposite handedness in rigid/similarity fits")
    parser.add_argument("--position-only", action="store_true", help="Fit old XYZ-only CSVs without orientation")
    parser.add_argument("--fit-offset", action="store_true",
                        help="Fit a rotating controller-to-flange position correction in Piper flange coordinates")
    parser.add_argument("--weighting", choices=("equal", "samples"), default="equal",
                        help="Equal weight per CSV (default), or equal weight per sample")
    args = parser.parse_args()
    try:
        result = fit_csvs(args.input, position_only=args.position_only, model=args.model,
                          allow_reflection=args.allow_reflection, weighting=args.weighting, fit_offset=args.fit_offset)
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if args.force else "x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print("Quest displacement -> Piper displacement matrix:")
    print(np.array(result["displacement_matrix_4x4"]))
    print(f"Position RMSE: {result['fit_errors']['position_rmse']:.6g} (Piper units)")
    if "orientation" in result:
        print(f"Orientation RMSE: {np.degrees(result['orientation']['fit_rmse_rad']):.6g} degrees")
    if args.fit_offset:
        print("Tracked-point correction (Piper flange frame):", result["tracked_point_offset"]["vector"])
        print(f"Position RMSE using mapped orientations: {result['fit_errors']['mapped_orientation_position_rmse']:.6g}")
        print(f"Position RMSE anchored at each recording start: {result['fit_errors']['anchored_pose_position_rmse']:.6g}")
    print(f"Saved {args.model} fit from {result['sample_count']} samples "
          f"across {result['recording_count']} CSV(s) to {output}")


if __name__ == "__main__":
    main()
