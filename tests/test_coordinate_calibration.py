import csv
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.calib.coordinate_record import CoordinateSession, build_parser, record, synchronize_positions
from scripts.calib.get_coord_transform import (
    fit_transform, load_positions, load_poses, fit_orientation_transform,
    fit_pose_transform, fit_csvs, map_relative_pose, transform_chunk,
)


class CoordinateCalibrationTests(unittest.TestCase):
    def rows(self, times, device, origin):
        fields = ("x", "y", "z") if device == "piper" else ("px", "py", "pz")
        return [dict(monotonic_ns=origin + round(t * 1e9),
                     **dict(zip(fields, (2*t, -t, 3*t + 1)))) for t in times]

    def test_recorder_logs_raw_jumps_and_retains_samples(self):
        session = CoordinateSession(jump_threshold=.05)
        session.start_ns = 1_000_000_000
        with patch("builtins.print") as log:
            for index, x in enumerate((0., .05, .06, 1., .06)):
                session.append_quest(dict(monotonic_ns=session.start_ns + index * 20_000_000,
                                          px=x, py=0., pz=0.))
            session.append_piper(dict(monotonic_ns=session.start_ns, x=0., y=0., z=0.))
            session.append_piper(dict(monotonic_ns=session.start_ns + 20_000_000, x=0., y=.1, z=0.))
        self.assertEqual(session.jump_counts, {"quest": 2, "piper": 1})
        self.assertEqual(len(session.quest_rows), 5)
        self.assertEqual(len(session.piper_rows), 2)
        self.assertEqual(log.call_count, 3)
        first = log.call_args_list[0].args[0]
        self.assertIn("[quest] POSITION JUMP #1", first)
        self.assertIn("t=0.060s", first)
        self.assertIn("step=0.9400", first)
        self.assertIn("dt=0.0200s", first)
        self.assertTrue(log.call_args_list[0].kwargs["flush"])

    def test_alignment_uses_overlap_and_linear_interpolation(self):
        origin = 12345678901234567
        rows = synchronize_positions(self.rows([0, 1, 2, 3], "piper", origin),
                                     self.rows([0.5, 1.5, 2.5], "quest", origin),
                                     origin_ns=origin, rate_hz=4, max_gap_s=1.1)
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[0]["monotonic_ns"], origin + 500000000)
        for row in rows:
            t = float(row["elapsed_s"])
            np.testing.assert_allclose([row[f"piper_{a}"] for a in "xyz"], [2*t, -t, 3*t+1])
            for axis in "xyz":
                self.assertAlmostEqual(row[f"piper_{axis}"], row[f"quest_{axis}"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coords.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            times, quest, piper = load_positions(path)
            np.testing.assert_allclose(quest, piper)
            self.assertEqual(len(times), len(rows))

    def test_gaps_are_excluded(self):
        rows = synchronize_positions(self.rows([0, .1, .2, .3, .4], "piper", 0),
                                     self.rows([0, .1, .4], "quest", 0),
                                     origin_ns=0, rate_hz=10, max_gap_s=.15)
        self.assertEqual([round(float(row["elapsed_s"]), 2) for row in rows], [0, .1, .4])

    def test_no_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            synchronize_positions(self.rows([0, 1], "piper", 0),
                                  self.rows([2, 3], "quest", 0), origin_ns=0)

    def test_known_relative_rotation_scale_and_linear_maps(self):
        quest = np.random.default_rng(1).normal(size=(100, 3))
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        for model, a in (("rigid", rotation), ("similarity", 2.5 * rotation),
                         ("linear", np.array([[1, .2, 0], [0, 2, 0], [.1, 0, 3]]))):
            with self.subTest(model=model):
                piper = quest @ a.T + [12, -3, 4]
                result = fit_transform(quest, piper, model=model)
                np.testing.assert_allclose(result["linear_matrix"], a, atol=1e-12)
                np.testing.assert_allclose(result["position_offset"], [12, -3, 4], atol=1e-12)
                np.testing.assert_allclose(np.array(result["displacement_matrix_4x4"])[:3, 3], 0)
                self.assertLess(result["fit_errors"]["position_rmse"], 1e-12)
                shifted = fit_transform(quest + 50, piper - 20, model=model)
                np.testing.assert_allclose(shifted["linear_matrix"], a, atol=1e-12)

    def test_reflection_is_explicit(self):
        quest = np.random.default_rng(2).normal(size=(100, 3))
        piper = quest * [1, -1, 1]
        proper = fit_transform(quest, piper)
        reflected = fit_transform(quest, piper, allow_reflection=True)
        self.assertGreater(proper["determinant"], 0)
        self.assertGreater(proper["fit_errors"]["position_rmse"], .1)
        self.assertLess(reflected["determinant"], 0)
        self.assertLess(reflected["fit_errors"]["position_rmse"], 1e-12)

    def test_record_worker_lifecycle_and_csv(self):
        def worker(device):
            def run(config, shared, recorder, *, init_done):
                init_done.set()
                while recorder.current() is None and not shared.shutdown.wait(.001):
                    pass
                session = recorder.current()
                if session is None:
                    return
                fields = ("x", "y", "z") if device == "piper" else ("px", "py", "pz")
                append = session.append_piper if device == "piper" else session.append_quest
                for _ in range(6):
                    stamp = time.monotonic_ns()
                    elapsed = session.elapsed_s(stamp)
                    row = dict(monotonic_ns=stamp, **dict(zip(fields, (elapsed, 2*elapsed, 3*elapsed))))
                    if device == "piper":
                        row.update(roll=0., pitch=0., yaw=elapsed)
                    else:
                        row.update(zip(("qx", "qy", "qz", "qw"), Rotation.from_euler("z", elapsed).as_quat()))
                    append(row)
                    shared.shutdown.wait(.005)
                shared.shutdown.wait()
            return run

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "coords.csv"
            args = build_parser().parse_args(["--output", str(output), "--duration", ".06", "--rate-hz", "200"])
            with (patch("scripts.calib.coordinate_record.run_piper_worker", worker("piper")),
                  patch("scripts.calib.coordinate_record.run_quest_worker", worker("quest")),
                  patch("scripts.calib.coordinate_record.FootPedalTrigger") as trigger):
                record(args)
                trigger.assert_not_called()
            times, quest, piper, qq, pq = load_poses(output)
            np.testing.assert_allclose(Rotation.from_quat(qq).as_matrix(), Rotation.from_quat(pq).as_matrix(), atol=1e-9)
            self.assertGreaterEqual(len(times), 4)
            np.testing.assert_allclose(quest, piper, atol=1e-10)

    def test_worker_failure_is_reported(self):
        def fail(*args, **kwargs):
            raise RuntimeError("no device")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "coords.csv"
            args = build_parser().parse_args(["--output", str(output)])
            with (patch("scripts.calib.coordinate_record.run_piper_worker", fail),
                  patch("scripts.calib.coordinate_record.run_quest_worker", fail)):
                with self.assertRaisesRegex(RuntimeError, "no device"):
                    record(args)
            self.assertFalse(output.exists())

    def test_pedal_waits_to_start_then_saves_and_releases_device(self):
        for interrupt_before_start in (False, True):
            with self.subTest(interrupt_before_start=interrupt_before_start), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "coords.csv"
                args = build_parser().parse_args(["--output", str(output)])
                managers = []

                def worker(config, shared, recorder, *, init_done):
                    managers.append(recorder)
                    init_done.set()
                    shared.shutdown.wait()

                polls = 0

                def poll():
                    nonlocal polls
                    polls += 1
                    manager = managers[0]
                    if polls <= 2:
                        self.assertIsNone(manager.current())
                        if interrupt_before_start:
                            raise KeyboardInterrupt
                        return [] if polls == 1 else [object()]
                    session = manager.current()
                    self.assertIsNotNone(session)
                    for i in range(6):
                        stamp = session.start_ns + i * 20_000_000
                        session.append_piper(dict(monotonic_ns=stamp, x=0., y=0., z=0.,
                                                  roll=0., pitch=0., yaw=0.))
                        session.append_quest(dict(monotonic_ns=stamp, px=0., py=0., pz=0.,
                                                  qx=0., qy=0., qz=0., qw=1.))
                    return [object()]

                with (patch("scripts.calib.coordinate_record.run_piper_worker", worker),
                      patch("scripts.calib.coordinate_record.run_quest_worker", worker),
                      patch("scripts.calib.coordinate_record.FootPedalTrigger") as trigger):
                    trigger.return_value.poll.side_effect = poll
                    record(args)
                    trigger.return_value.start.assert_called_once()
                    trigger.return_value.stop.assert_called_once()
                self.assertIsNone(managers[0].current())
                if interrupt_before_start:
                    self.assertFalse(output.exists())
                else:
                    times, *_ = load_poses(output)
                    self.assertEqual(len(times), 6)

    def test_slerp_crosses_euler_wrap_and_ignores_quaternion_sign(self):
        piper = self.rows([0, 1], "piper", 0)
        quest = self.rows([0, 1], "quest", 0)
        for i, yaw in enumerate((170., -170.)):
            piper[i].update(roll=0., pitch=0., yaw=np.radians(yaw))
            quat = Rotation.from_euler("z", yaw, degrees=True).as_quat() * (-1 if i else 1)
            quest[i].update(zip(("qx", "qy", "qz", "qw"), quat))
        rows = synchronize_positions(piper, quest, origin_ns=0, rate_hz=2,
                                     max_gap_s=1.1, include_orientation=True)
        for prefix in ("piper", "quest"):
            middle = Rotation.from_quat([rows[1][f"{prefix}_{key}"] for key in ("qx", "qy", "qz", "qw")])
            np.testing.assert_allclose(middle.as_matrix(), np.diag([-1., -1., 1.]), atol=1e-12)
        quest[0].update(qx=0, qy=0, qz=0, qw=0)
        with self.assertRaises(ValueError):
            synchronize_positions(piper, quest, origin_ns=0, rate_hz=2,
                                  max_gap_s=1.1, include_orientation=True)

    def pose_fixture(self):
        rng = np.random.default_rng(13)
        quest = rng.normal(size=(80, 3))
        rq = Rotation.from_rotvec(rng.normal(scale=.3, size=(80, 3))).as_matrix()
        b = Rotation.from_euler("xyz", [.4, -.2, .7]).as_matrix()
        # Translation basis is deliberately different from the angular basis.
        a = np.array([[0., -2., 0.], [2., 0., 0.], [0., 0., 2.]])
        piper = quest @ a.T + [4., -3., 2.]
        tool_offset = Rotation.from_euler("xyz", [.6, .2, -.3]).as_matrix()
        rp = b @ rq @ tool_offset
        qq, pq = Rotation.from_matrix(rq).as_quat(), Rotation.from_matrix(rp).as_quat()
        calibration = fit_pose_transform(quest, piper, qq, pq, model="similarity")
        return quest, piper, rq, rp, qq, pq, calibration, b

    def test_relative_orientation_fit_cancels_tool_offsets_and_quaternion_signs(self):
        quest, piper, rq, rp, qq, pq, result, b = self.pose_fixture()
        np.testing.assert_allclose(result["orientation"]["rotation_matrix"], b, atol=1e-12)
        self.assertLess(result["orientation"]["fit_rmse_rad"], 1e-12)
        qq[::2] *= -1
        pq[1::2] *= -1
        result2 = fit_orientation_transform(qq, pq)
        np.testing.assert_allclose(result2["rotation_matrix"], b, atol=1e-12)
        offset = Rotation.from_euler("zyx", [.2, -.3, .4]).as_matrix()
        result3 = fit_orientation_transform(Rotation.from_matrix(rq @ offset).as_quat(), pq)
        np.testing.assert_allclose(result3["rotation_matrix"], b, atol=1e-12)

    def test_fit_offset_recovers_mapping_and_converts_world_and_chunk_deltas(self):
        q, _, rq, rp, qq, pq, _, _ = self.pose_fixture()
        offset = np.array([-.09, .03, -.04])
        for model in ("rigid", "similarity", "linear"):
            for reflected in (False, True):
                with self.subTest(model=model, reflected=reflected):
                    a = Rotation.from_euler("xyz", [.2, -.5, .3]).as_matrix()
                    if reflected:
                        a = a @ np.diag([-1., 1., 1.])
                    if model == "similarity":
                        a *= .97
                    elif model == "linear":
                        a = a @ np.array([[.9, .1, 0.], [0., 1.1, 0.], [0., 0., .8]])
                    p = q @ a.T + rp @ offset + [4., -3., 2.]
                    # Different recording origins and fixed orientation offsets
                    # must not be mistaken for displacement at the boundary.
                    p[40:] += [20., 30., -40.]
                    result = fit_pose_transform(q, p, qq, pq, model=model,
                                                allow_reflection=reflected, fit_offset=True,
                                                recording_lengths=[40, 40])
                    np.testing.assert_allclose(result["linear_matrix"], a, atol=1e-7)
                    np.testing.assert_allclose(result["tracked_point_offset"]["vector"], offset, atol=1e-7)
                    self.assertLess(result["fit_errors"]["position_rmse"], 1e-8)
                    self.assertLess(result["fit_errors"]["anchored_pose_position_rmse"], 1e-7)
                    self.assertIsNone(result["position_matrix_4x4"])
                    start, end = 7, 25
                    targets, _ = transform_chunk(q[start:end], qq[start:end], p[start], pq[start], result)
                    np.testing.assert_allclose(targets, p[start:end], atol=1e-7)
                    world_dp = q[start:end] - q[start]
                    world_dq = Rotation.from_matrix(rq[start:end] @ rq[start].T).as_quat()
                    dp, _ = map_relative_pose(world_dp, world_dq, result, piper_start_quaternion=pq[start])
                    np.testing.assert_allclose(dp, p[start:end] - p[start], atol=1e-7)
                    local_dp = world_dp @ rq[start]
                    local_dq = Rotation.from_matrix(rq[start].T @ rq[start:end]).as_quat()
                    dp, _ = map_relative_pose(local_dp, local_dq, result, relative_frame="chunk",
                                              quest_start_quaternion=qq[start], piper_start_quaternion=pq[start])
                    np.testing.assert_allclose(dp, (p[start:end] - p[start]) @ rp[start], atol=1e-7)
                    single, _ = map_relative_pose(world_dp[-1], world_dq[-1], result,
                                                  piper_start_quaternion=pq[start])
                    np.testing.assert_allclose(single, p[end - 1] - p[start], atol=1e-7)
                    with self.assertRaisesRegex(ValueError, "piper_start_quaternion"):
                        map_relative_pose(world_dp, world_dq, result)

    def test_offset_fit_requires_observable_rotation_and_pose_data(self):
        q, p, _, _, qq, _, _, _ = self.pose_fixture()
        fixed = np.tile([0., 0., 0., 1.], (len(q), 1))
        with self.assertRaisesRegex(ValueError, "underdetermined"):
            fit_pose_transform(q, p, qq, fixed, fit_offset=True)
        with self.assertRaisesRegex(ValueError, "omit --position-only"):
            fit_csvs(["unused.csv"], position_only=True, fit_offset=True)

    def test_arbitrary_chunk_starts_recover_absolute_piper_poses(self):
        quest, piper, rq, rp, qq, pq, calibration, _ = self.pose_fixture()
        for start in (0, 7, 31):
            positions, quats = transform_chunk(quest[start:start+10], qq[start:start+10],
                                              piper[start], pq[start], calibration)
            np.testing.assert_allclose(positions, piper[start:start+10], atol=1e-12)
            np.testing.assert_allclose(Rotation.from_quat(quats).as_matrix(), rp[start:start+10], atol=1e-12)
        # Changing absolute Quest origin cannot change chunk targets; Piper anchors are fresh.
        targets, quats = transform_chunk(quest + [20, -10, 4], qq, [3, 5, 7], [0, 0, 0, 1], calibration)
        np.testing.assert_allclose(targets[0], [3, 5, 7], atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(quats[0]).as_matrix(), np.eye(3), atol=1e-12)
        expected = np.array([3, 5, 7]) + (quest - quest[0]) @ np.array(calibration["linear_matrix"]).T
        np.testing.assert_allclose(targets, expected, atol=1e-12)

    def test_chunk_local_deltas_account_for_both_start_orientations(self):
        quest, piper, rq, rp, qq, pq, calibration, _ = self.pose_fixture()
        start = 7
        local_dp = (quest[start:] - quest[start]) @ rq[start]
        local_dr = rq[start].T @ rq[start:]
        dp, dq = map_relative_pose(local_dp, Rotation.from_matrix(local_dr).as_quat(), calibration,
                                  relative_frame="chunk", quest_start_quaternion=qq[start],
                                  piper_start_quaternion=pq[start])
        np.testing.assert_allclose(dp, (piper[start:] - piper[start]) @ rp[start], atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(dq).as_matrix(), rp[start].T @ rp[start:], atol=1e-12)
        # Also accept one relative pose rather than a batch.
        one_dp, one_dq = map_relative_pose(local_dp[2], Rotation.from_matrix(local_dr[2]).as_quat(), calibration,
                                         relative_frame="chunk", quest_start_quaternion=qq[start],
                                         piper_start_quaternion=pq[start])
        np.testing.assert_allclose(one_dp, dp[2], atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(one_dq).as_matrix(), rp[start].T @ rp[start+2], atol=1e-12)
        with self.assertRaisesRegex(ValueError, "starting orientations"):
            map_relative_pose(local_dp, Rotation.from_matrix(local_dr).as_quat(), calibration, relative_frame="chunk")

    def write_pose_csv(self, path, quest, piper, qq=None, pq=None):
        fields = ["elapsed_s"] + [f"{device}_{axis}" for device in ("quest", "piper") for axis in "xyz"]
        if qq is not None:
            fields += [f"{device}_{key}" for device in ("quest", "piper") for key in ("qx", "qy", "qz", "qw")]
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for i, (q, p) in enumerate(zip(quest, piper)):
                row = [i / 50, *q, *p]
                if qq is not None:
                    row += [*qq[i], *pq[i]]
                writer.writerow(row)

    def test_multiple_csvs_use_independent_position_and_orientation_references(self):
        q, p, rq, rp, qq, pq, expected, b = self.pose_fixture()
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.csv", Path(directory) / "second.csv"
            self.write_pose_csv(first, q, p, qq, pq)
            tool = Rotation.from_euler("xyz", [.7, -.4, .2]).as_matrix()
            self.write_pose_csv(second, q[17:] + [40, 50, 60], p[17:] - [100, 200, 300],
                                Rotation.from_matrix(rq[17:] @ tool).as_quat(), pq[17:])
            result = fit_csvs([first, second], model="similarity")
            np.testing.assert_allclose(result["linear_matrix"], expected["linear_matrix"], atol=1e-12)
            np.testing.assert_allclose(result["orientation"]["rotation_matrix"], b, atol=1e-12)
            self.assertEqual(result["sample_count"], len(q) + len(q[17:]))
            self.assertEqual(result["recording_count"], 2)
            self.assertIsNone(result["position_matrix_4x4"])
            self.assertLess(result["fit_errors"]["consecutive_displacement_rmse"], 1e-12)
            self.assertEqual(result["recordings"][1]["input_csv"], str(second))
            single = fit_csvs([first], model="similarity")
            np.testing.assert_allclose(single["position_matrix_4x4"], expected["position_matrix_4x4"], atol=1e-12)
            for recording in result["recordings"]:
                self.assertLess(recording["orientation_errors"]["fit_rmse_rad"], 1e-12)

    def test_multiple_csv_weighting_is_equal_per_file_by_default(self):
        q = np.random.default_rng(11).normal(size=(30, 3))
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.csv", Path(directory) / "second.csv"
            self.write_pose_csv(first, q, q)
            long_q = np.tile(q, (5, 1))
            self.write_pose_csv(second, long_q + 100, 3 * long_q - 200)
            equal = fit_csvs([first, second], position_only=True, model="similarity")
            samples = fit_csvs([first, second], position_only=True, model="similarity", weighting="samples")
            self.assertAlmostEqual(equal["uniform_scale"], 2.)
            self.assertAlmostEqual(samples["uniform_scale"], 16 / 6)
            self.assertNotIn("orientation", equal)

    def test_nonfinite_quaternions_are_rejected(self):
        quats = np.tile([0., 0., 0., 1.], (20, 1))
        quats[1, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_orientation_transform(quats, quats)

    def test_orientation_requires_multiple_axes(self):
        identity = np.tile([0., 0., 0., 1.], (20, 1))
        for quats in (identity, Rotation.from_rotvec(np.column_stack((np.zeros((20, 2)), np.linspace(0, 1, 20)))).as_quat()):
            with self.assertRaisesRegex(ValueError, "nonparallel axes"):
                fit_orientation_transform(quats, quats)

    def test_degenerate_or_invalid_motion_fails(self):
        for quest in (np.zeros((10, 3)), np.tile(np.arange(10)[:, None], (1, 3)),
                      np.column_stack((np.eye(4, 2), np.zeros(4)))):
            with self.assertRaisesRegex(ValueError, "three axes"):
                fit_transform(quest, quest)
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_transform(np.full((10, 3), np.nan), np.ones((10, 3)))


if __name__ == "__main__":
    unittest.main()
