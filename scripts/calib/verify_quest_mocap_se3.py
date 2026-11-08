#!/usr/bin/env python3
"""Non-production synthetic audit of A X = Y B for mocap/Quest calibration.

Run with a Python providing cv2.calibrateHandEye (system Python here):
    python3 scripts/calib/verify_quest_mocap_se3.py --seeds 8

Writes reproducible synthetic results only; never reads hardware or changes
production calibration. All poses are active, column-vector, local-to-world,
right-handed SE(3), in meters. Unity conversion is explicit and tested separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
P = np.diag([1., 1., -1.])
H = np.eye(4)
H[:3, :3] = P
SIGMA_T = .002
SIGMA_R = np.deg2rad(.3)


def pose(rotation, translation):
    rotation, translation = np.asarray(rotation), np.asarray(translation)
    t = np.broadcast_to(np.eye(4), rotation.shape[:-2] + (4, 4)).copy()
    t[..., :3, :3], t[..., :3, 3] = rotation, translation
    return t


def inv(t):
    r = np.swapaxes(t[..., :3, :3], -1, -2)
    return pose(r, -(r @ t[..., :3, 3, None])[..., 0])


def valid(t):
    return (np.isfinite(t).all() and
            np.allclose(np.swapaxes(t[..., :3, :3], -1, -2) @ t[..., :3, :3], np.eye(3), atol=1e-6) and
            np.allclose(np.linalg.det(t[..., :3, :3]), 1., atol=1e-6) and
            np.allclose(t[..., 3, :], [0, 0, 0, 1]))


def error(truth, estimate):
    delta = inv(truth) @ estimate
    return {'translation_m': float(np.linalg.norm(delta[:3, 3])),
            'rotation_deg': float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude()))}


def metrics(a, b, x, y):
    prediction = y @ b @ inv(x)
    delta = inv(a) @ prediction
    translation = np.linalg.norm(delta[:, :3, 3], axis=1)
    angle = np.rad2deg(Rotation.from_matrix(delta[:, :3, :3]).magnitude())
    return {'translation_rmse_m': float(np.sqrt(np.mean(translation ** 2))),
            'translation_mae_m': float(np.mean(translation)),
            'translation_p95_m': float(np.percentile(translation, 95)),
            'translation_max_m': float(np.max(translation)),
            'rotation_rmse_deg': float(np.sqrt(np.mean(angle ** 2))),
            'rotation_p95_deg': float(np.percentile(angle, 95)),
            'rotation_max_deg': float(np.max(angle))}


def handeye(a, b, method):
    # gripper=R, base=M, camera=C, target=Q; inv(B) maps Q->C.
    bi = inv(b)
    r, t = cv2.calibrateHandEye(list(a[:, :3, :3]), list(a[:, :3, 3]),
                              list(bi[:, :3, :3]), list(bi[:, :3, 3]), method=method)
    x = pose(r, np.asarray(t).reshape(3))
    if not valid(x):
        raise ValueError('OpenCV returned a nonfinite or non-SE(3) transform')
    yi = a @ x @ bi
    # SO(3) chordal/Frechet embedding mean, not elementwise or Euler averaging.
    y = pose(Rotation.from_matrix(yi[:, :3, :3]).mean().as_matrix(), np.mean(yi[:, :3, 3], axis=0))
    return x, y


def robotworld(a, b, method):
    # OpenCV equation A_cv X_cv = Z_cv B_cv. Set A_cv=A, B_cv=B:
    # camera=M, world=R, gripper=Q, base=C, outputs X and Y directly.
    rx, tx, ry, ty = cv2.calibrateRobotWorldHandEye(
        list(a[:, :3, :3]), list(a[:, :3, 3]), list(b[:, :3, :3]), list(b[:, :3, 3]), method=method)
    x, y = pose(rx, np.asarray(tx).reshape(3)), pose(ry, np.asarray(ty).reshape(3))
    if not valid(x) or not valid(y):
        raise ValueError('OpenCV returned a nonfinite or non-SE(3) transform')
    return x, y


def residual(a, b, x, y):
    e = inv(a @ x) @ (y @ b)
    return np.column_stack((e[:, :3, 3] / SIGMA_T,
                            Rotation.from_matrix(e[:, :3, :3]).as_rotvec() / SIGMA_R))


def perturb(base, v):
    # Rotation updates stay on SO(3); translation updates are in the destination frame.
    return pose(Rotation.from_rotvec(v[:3]).as_matrix() @ base[:3, :3], base[:3, 3] + v[3:])


def motion_guard(a):
    if len(a) < 3:
        raise ValueError('At least 3 poses required')
    relative = inv(a[0]) @ a
    vectors = Rotation.from_matrix(relative[:, :3, :3]).as_rotvec()
    s = np.linalg.svd(vectors, compute_uv=False) / np.sqrt(len(a))
    if s[1] < .05 or s[1] < .05 * s[0]:
        raise ValueError('Insufficient nonparallel rotational excitation; full offset is not reliable')
    return s


def refine(a, b, x, y, robust=False):
    weights = np.ones(len(a))
    # Pose-block Cauchy IRLS, so robust weighting is invariant to rotating XYZ axes.
    for _ in range(8 if robust else 1):
        def objective(v):
            return (residual(a, b, perturb(x, v[:6]), perturb(y, v[6:])) * np.sqrt(weights[:, None])).ravel()
        result = least_squares(objective, np.zeros(12), x_scale='jac', max_nfev=180,
                               ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if not result.success:
            raise ValueError('Joint refinement did not converge')
        x, y = perturb(x, result.x[:6]), perturb(y, result.x[6:])
        if robust:
            norm = np.linalg.norm(residual(a, b, x, y), axis=1)
            weights = 1 / (1 + (norm / 4.) ** 2)
    return x, y


def robust_fit(a, b, rng):
    motion_guard(a)
    best = None
    # Sample whole poses, not all correlated relative-motion pairs.
    for _ in range(50):
        ids = rng.choice(len(a), 5, replace=False)
        try:
            motion_guard(a[ids])
            x, y = handeye(a[ids], b[ids], cv2.CALIB_HAND_EYE_PARK)
            norms = np.linalg.norm(residual(a, b, x, y), axis=1)
            score = np.median(norms)
            if best is None or score < best[0]:
                best = score, x, y, norms
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            continue
    if best is None:
        raise ValueError('No valid robust initialization')
    _, x, y, norms = best
    mask = norms < max(5., 2.5 * np.median(norms))
    if mask.sum() < 6:
        raise ValueError('Insufficient consensus')
    x, y = refine(a[mask], b[mask], x, y)
    x, y = refine(a, b, x, y, robust=True)
    spectrum = jacobian_spectrum(a[mask], b[mask], x, y)
    if spectrum['rank_relative_1e-6'] != 12 or spectrum['condition'] > 1e4:
        raise ValueError('Joint calibration is rank deficient or poorly conditioned')
    return x, y


def random_pose(rng, translation_scale=.3):
    return pose(Rotation.random(random_state=rng).as_matrix(), rng.normal(size=3) * translation_scale)


def make_data(rng, mode='diverse', offset_m=.12, n=90):
    x, y = random_pose(rng), random_pose(rng, .7)
    x[:3, 3] = rng.normal(size=3)
    x[:3, 3] *= offset_m / np.linalg.norm(x[:3, 3])
    if mode in ('diverse', 'rotation_only'):
        rotation = Rotation.random(n, random_state=rng).as_matrix()
    elif mode == 'translation_only':
        rotation = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    elif mode in ('single_axis', 'single_axis_translations'):
        vectors = np.zeros((n, 3)); vectors[:, 2] = rng.uniform(-1.2, 1.2, n)
        rotation = Rotation.from_rotvec(vectors).as_matrix()
    elif mode == 'nearly_degenerate':
        vectors = rng.normal(scale=1e-4, size=(n, 3)); vectors[:, 2] += rng.uniform(-1.2, 1.2, n)
        rotation = Rotation.from_rotvec(vectors).as_matrix()
    else:
        raise ValueError(mode)
    translation = rng.normal(size=(n, 3)) * .25
    if mode in ('rotation_only', 'single_axis'):
        translation[:] = [.2, -.1, .4]
    b = pose(rotation, translation)
    a = y @ b @ inv(x)
    np.testing.assert_allclose(a @ x, y @ b, atol=1e-13)
    return a, b, x, y


def noise(t, rng, translation_sigma=0., rotation_sigma=0., outliers=False):
    out = t.copy()
    out[:, :3, 3] += rng.normal(scale=translation_sigma, size=(len(t), 3))
    dr = Rotation.from_rotvec(rng.normal(scale=rotation_sigma, size=(len(t), 3))).as_matrix()
    out[:, :3, :3] = dr @ out[:, :3, :3]
    if outliers:
        ids = rng.choice(len(t), 6, replace=False)
        out[ids, :3, 3] += rng.normal(scale=.08, size=(len(ids), 3))
        out[ids, :3, :3] = Rotation.from_rotvec(rng.normal(scale=.25, size=(len(ids), 3))).as_matrix() @ out[ids, :3, :3]
    return out


def jacobian_spectrum(a, b, x, y):
    base = residual(a, b, x, y).ravel()
    columns = []
    for j in range(12):
        v = np.zeros(12); v[j] = 1e-6
        columns.append((residual(a, b, perturb(x, v[:6]), perturb(y, v[6:])).ravel() - base) / 1e-6)
    s = np.linalg.svd(np.array(columns).T, compute_uv=False)
    return {'singular_values': s.tolist(), 'rank_relative_1e-6': int(np.sum(s > s[0] * 1e-6)),
            'condition': float(s[0] / max(s[-1], 1e-15))}


def conversion_checks():
    checks = 0
    for axis in range(3):
        v = np.eye(3)[axis] * np.pi / 2
        ru = Rotation.from_rotvec(v).as_matrix()
        converted = P @ ru @ P.T
        q = Rotation.from_matrix(ru).as_quat()
        mapped = np.r_[np.linalg.det(P) * P @ q[:3], q[3]]
        np.testing.assert_allclose(Rotation.from_quat(mapped).as_matrix(), converted, atol=1e-14)
        np.testing.assert_allclose(converted @ (P @ np.eye(3)), P @ ru, atol=1e-14)
        assert abs(np.linalg.det(converted) - 1) < 1e-14
        # Independent expected 90-degree axis sign for reflection Z.
        expected = Rotation.from_rotvec(np.eye(3)[axis] * (1 if axis == 2 else -1) * np.pi/2).as_matrix()
        np.testing.assert_allclose(converted, expected, atol=1e-14)
        checks += 1
    rng = np.random.default_rng(91)
    a, b, x, y = make_data(rng)
    unity = H @ b @ H
    recovered = H @ unity @ H
    np.testing.assert_allclose(recovered, b, atol=1e-14)
    xr, yr = handeye(a, recovered, cv2.CALIB_HAND_EYE_PARK)
    assert error(x, xr)['rotation_deg'] < 1e-7
    # One-sided reflection is not an SE(3) pose and must be rejected.
    assert not valid(H @ unity)
    # Original mixed-handed X/Y are improper maps, not solver-ready SE(3).
    np.testing.assert_allclose(a @ (x @ H), (y @ H) @ unity, atol=1e-13)
    # Meters/mm conversion round-trip and quaternion sign invariance.
    np.testing.assert_allclose(a[:, :3, 3] * 1000 / 1000, a[:, :3, 3], atol=1e-14)
    q = Rotation.from_matrix(b[:, :3, :3]).as_quat()
    np.testing.assert_allclose(Rotation.from_quat(q).as_matrix(), Rotation.from_quat(-q).as_matrix(), atol=1e-14)
    # Relabel all FOUR frames independently; verify the corresponding X,Y
    # transformation law rather than relying on particular world orientations.
    km, kr, kq, kc = [random_pose(rng) for _ in range(4)]
    anew, bnew = km @ a @ inv(kr), kq @ b @ inv(kc)
    xnew, ynew = handeye(anew, bnew, cv2.CALIB_HAND_EYE_PARK)
    frame_errors = {'X': error(kr @ x @ inv(kc), xnew),
                    'Y': error(km @ y @ inv(kq), ynew)}
    for e in frame_errors.values():
        assert e['translation_m'] < 1e-7 and e['rotation_deg'] < 1e-7
    return {'known_90_degree_axes_passed': checks, 'unity_roundtrip_calibration': {'X': error(x, xr), 'Y': error(y, yr)},
            'one_sided_reflection_rejected': True, 'mm_roundtrip': True, 'quaternion_sign_invariance': True,
            'independent_relabeling_of_all_four_frames': frame_errors}


def production_review():
    from scripts.quest.visualize_quest_accuracy import fit_offset_alignment
    rows = []
    for mode in ('diverse', 'rotation_only'):
        rng = np.random.default_rng(42)
        a, b, x, y = make_data(rng, mode=mode)
        # Recover raw Unity positions; current viewer accepts mixed-handed data.
        unity_positions = (H @ b @ H)[:, :3, 3]
        try:
            matrix, offset, body_offset, rms = fit_offset_alignment(unity_positions, a[:, :3, 3], a[:, :3, :3])
            rows.append({'case': mode, 'status': 'accepted', 'rms_m': rms,
                         'world_matrix_error': float(np.linalg.norm(matrix - y[:3, :3] @ P)),
                         'world_translation_error_m': float(np.linalg.norm(offset-y[:3, 3])),
                         'body_offset_error_m': float(np.linalg.norm(body_offset-x[:3, 3]))})
            assert rms < 1e-7
        except ValueError as exc:
            rows.append({'case': mode, 'status': 'rejected', 'reason': str(exc)})
    # Existing robust fit eliminates translation with ordinary means, which is
    # exact for least squares but not the same optimum under a robust loss.
    rng = np.random.default_rng(42)
    a, b, x, y = make_data(rng)
    corrupted = a.copy()
    corrupted[:6, :3, 3] += [.12, .06, -.08]
    matrix, offset, body, rms = fit_offset_alignment((H @ b @ H)[:, :3, 3],
                                                    corrupted[:, :3, 3], corrupted[:, :3, :3])
    validation_b = pose(Rotation.random(100, random_state=rng).as_matrix(), rng.normal(size=(100, 3)) * .25)
    validation_a = y @ validation_b @ inv(x)
    raw_q = (H @ validation_b @ H)[:, :3, 3]
    errors = raw_q @ matrix.T + offset - validation_a[:, :3, 3] - validation_a[:, :3, :3] @ body
    rows.append({'case': 'position_only_positive_outlier_bias',
                 'six_corrupted_positions_m': [.12, .06, -.08],
                 'world_translation_error_m': float(np.linalg.norm(offset-y[:3, 3])),
                 'body_offset_error_m': float(np.linalg.norm(body-x[:3, 3])),
                 'heldout_position_rmse_m': float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
                 'training_rms_m': rms})
    # Probe additional existing Vicon/FK full-pose fitter without changing it.
    from scripts.calib.fit_quest_calibration import fit_vicon_to_flange, vicon_flange
    rng = np.random.default_rng(9)
    a, b, x, y = make_data(rng)
    # It fits B = W A X, hence W=inv(Y), body=X for these inputs.
    transform = fit_vicon_to_flange(a[:, :3, 3], a[:, :3, :3], b[:, :3, 3], b[:, :3, :3])
    rw, tw, rx, tx = transform
    rows.append({'case': 'existing_fit_vicon_to_flange', 'world': error(inv(y), pose(rw, tw)),
                 'body': error(x, pose(rx, tx))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--output', type=Path, default=ROOT/'output/quest_se3_verification/results.json')
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    if not hasattr(cv2, 'calibrateHandEye') or not hasattr(cv2, 'calibrateRobotWorldHandEye'):
        parser.error(f'OpenCV {cv2.__version__} lacks hand-eye bindings; use system python3 (4.13.0 in this workspace)')
    methods = {name: (lambda a, b, method=getattr(cv2, 'CALIB_HAND_EYE_' + name): handeye(a, b, method))
               for name in ('TSAI', 'PARK', 'HORAUD', 'ANDREFF', 'DANIILIDIS')}
    methods.update({name: (lambda a, b, method=getattr(cv2, 'CALIB_ROBOT_WORLD_HAND_EYE_' + name): robotworld(a, b, method))
                    for name in ('SHAH', 'LI')})
    methods['PARK_REFINE'] = lambda a, b: refine(a, b, *handeye(a, b, cv2.CALIB_HAND_EYE_PARK))
    cases = [('noise_free', 'diverse', 0., 0., False, .12),
             ('translation_noise', 'diverse', .001, 0., False, .12),
             ('rotation_noise', 'diverse', 0., np.deg2rad(.2), False, .12),
             ('combined_noise', 'diverse', .001, np.deg2rad(.2), False, .12),
             ('outliers', 'diverse', .001, np.deg2rad(.2), True, .12),
             ('zero_offset', 'diverse', 0., 0., False, 0.),
             ('large_offset', 'diverse', 0., 0., False, .5),
             ('rotation_only', 'rotation_only', 0., 0., False, .12),
             ('nearly_degenerate', 'nearly_degenerate', .001, np.deg2rad(.2), False, .12),
             ('translation_only', 'translation_only', 0., 0., False, .12),
             ('single_axis', 'single_axis', 0., 0., False, .12),
             ('single_axis_translations', 'single_axis_translations', 0., 0., False, .12)]
    results = []
    representative = None
    spectra = {}
    for case, mode, st, sr, outliers, offset_m in cases:
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed + 210)
            a0, b0, x_true, y_true = make_data(rng, mode, offset_m)
            nfit = 60
            a, b = noise(a0[:nfit], rng, st, sr, outliers), noise(b0[:nfit], rng, st, sr)
            a_test, b_test = noise(a0[nfit:], rng, st, sr), noise(b0[nfit:], rng, st, sr)
            b_diverse = pose(Rotation.random(30, random_state=rng).as_matrix(), rng.normal(size=(30, 3)) * .25)
            a_diverse = y_true @ b_diverse @ inv(x_true)
            if seed == 0:
                spectra[case] = jacobian_spectrum(a0[:nfit], b0[:nfit], x_true, y_true)
            try:
                motion_guard(a)
                guard = 'accepted'
            except ValueError as exc:
                guard = str(exc)
            for name, fit in list(methods.items()) + [('ROBUST_REFINE', lambda a, b: robust_fit(a, b, rng))]:
                row = {'case': case, 'seed': seed, 'method': name, 'motion_guard': guard}
                try:
                    # Raw OpenCV methods intentionally run without guard to expose failures.
                    x, y = fit(a, b)
                    row.update(status='accepted', X_error=error(x_true, x), Y_error=error(y_true, y),
                               training=metrics(a, b, x, y), heldout_clean=metrics(a0[nfit:], b0[nfit:], x, y),
                               heldout_noisy=metrics(a_test, b_test, x, y),
                               heldout_diverse=metrics(a_diverse, b_diverse, x, y))
                    if case in ('noise_free', 'zero_offset', 'large_offset', 'rotation_only') and name in ('PARK', 'SHAH', 'PARK_REFINE', 'ROBUST_REFINE'):
                        assert max(row['X_error']['translation_m'], row['Y_error']['translation_m']) < 1e-7
                        assert max(row['X_error']['rotation_deg'], row['Y_error']['rotation_deg']) < 1e-5
                    if case == 'noise_free' and seed == 0 and name == 'PARK':
                        representative = {k: v.tolist() for k, v in dict(X_true=x_true, X_estimated=x, Y_true=y_true, Y_estimated=y).items()}
                except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                    row.update(status='rejected', reason=str(exc))
                results.append(row)
        print(f'{case}: completed {args.seeds} seeds', flush=True)
    data = {'versions': dict(python=sys.version, opencv=cv2.__version__, numpy=np.__version__, scipy=scipy.__version__),
            'seed_count': args.seeds, 'training_poses': 60, 'heldout_poses': 30,
            'noise': 'independent destination-frame additive position and left rotational noise on both streams; outliers in six A poses only',
            'representative': representative, 'conversions': conversion_checks(), 'observability': spectra,
            'production_review': production_review(), 'runs': results}
    for case in ('nearly_degenerate', 'translation_only', 'single_axis', 'single_axis_translations'):
        rows = [r for r in results if r['case'] == case and r['method'] == 'ROBUST_REFINE']
        assert all(r['status'] == 'rejected' for r in rows), case
    # Verify the exact residual is zero at truth; wrong inversion fails even on paired poses.
    rng = np.random.default_rng(21)
    a, b, x, y = make_data(rng)
    assert np.max(np.abs(residual(a, b, x, y))) < 1e-10
    wrong = handeye(a, inv(b), cv2.CALIB_HAND_EYE_PARK)
    data['wrong_B_inversion_control'] = metrics(a, b, *wrong)
    assert data['wrong_B_inversion_control']['translation_rmse_m'] > .1
    assert data['wrong_B_inversion_control']['rotation_rmse_deg'] > 10
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
