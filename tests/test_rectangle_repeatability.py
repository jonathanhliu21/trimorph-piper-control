import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.calib.analyze_rectangle import compare_to_commanded, corner_stats, fit_rigid, resample
from scripts.calib.rectangle_repeatability import (
    DEFAULT_HOME_DEG, build_schedule, ik_track, joint_margins, parse_args, piper_fk, rectangle_corners,
    rectangle_joint_path, target_at,
)


class RectangleRepeatabilityTests(unittest.TestCase):
    def test_corners_start_at_pose_and_span_plane(self):
        corners = rectangle_corners([0.3, 0.0, 0.2], "xz", 0.02, 0.01)
        np.testing.assert_allclose(corners, [[0.3, 0, 0.2], [0.32, 0, 0.2], [0.32, 0, 0.21], [0.3, 0, 0.21]])

    def test_schedule_closes_each_loop_with_dwells(self):
        corners = rectangle_corners([0, 0, 0], "xy", 0.02, 0.02)
        schedule = build_schedule(corners, 2, 0.01, 0.5)
        self.assertEqual(len(schedule), 1 + 2 * 8)
        self.assertAlmostEqual(schedule[-1][1], 0.5 + 2 * 4 * (2.0 + 0.5))
        self.assertEqual([segment[5] for segment in schedule if segment[3] == "dwell"], [0, 1, 2, 3, 0, 1, 2, 3, 0])
        np.testing.assert_allclose(target_at(schedule[1], schedule[1][0] + 1.0), [0.01, 0, 0])
        np.testing.assert_allclose(target_at(schedule[1], 1e9), corners[1])

    def test_ik_follows_rectangle_from_start_joints(self):
        start = np.radians(DEFAULT_HOME_DEG)
        pose = piper_fk(start)
        target = np.concatenate([pose[:3] + [0.01, -0.005, 0.002], pose[3:]])
        np.testing.assert_allclose(piper_fk(ik_track(start, target))[:3], target[:3], atol=1e-6)
        corners = rectangle_corners(pose[:3], "xy", 0.02, 0.02)
        path = rectangle_joint_path(start, corners, pose[3:])
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(path[-1], start, atol=1e-4)  # closed loop returns to the same branch
        self.assertLess(np.degrees(np.max(np.abs(np.diff(path, axis=0)))), 2.0)

    def test_joint_margins_flag_limits_and_wrist(self):
        path = np.radians([[0, 90, -90, 0, 30, 0], [0, 90, -90, 0, 5, 115]])
        rows, wrist_deg, problems = joint_margins(path, 10, 15)
        self.assertAlmostEqual(rows[5]["margin_deg"], 5.0, places=3)  # limits are stored as rounded radians
        self.assertAlmostEqual(wrist_deg, 5.0)
        self.assertEqual(len(problems), 2)
        self.assertEqual(joint_margins(path[:1], 10, 15)[2], [])

    def test_start_options(self):
        self.assertEqual(list(parse_args([]).start_joints_deg), list(DEFAULT_HOME_DEG))
        self.assertTrue(parse_args(["--start-here"]).start_here)
        with self.assertRaises(SystemExit):
            parse_args(["--start-here", "--start-joints-deg", "0", "90", "-90", "0", "30", "0"])

    def test_rejects_large_rectangle(self):
        with self.assertRaises(SystemExit):
            parse_args(["--width-mm", "150"])

    def test_rigid_fit_recovers_planar_transform(self):
        grid = np.array([[x, y, 0.0] for x in range(5) for y in range(3)])
        rotation = Rotation.from_euler("zyx", [30, 10, -5], degrees=True).as_matrix()
        tracker = (grid - [1, 2, 3]) @ rotation  # tracker = R.T (grid - offset)
        fitted, translation = fit_rigid(tracker, grid)
        np.testing.assert_allclose(tracker @ fitted.T + translation, grid, atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(fitted), 1.0)

    def test_resample_marks_gaps(self):
        times = np.array([0.0, 0.01, 0.02, 0.5, 0.51])
        values = np.column_stack([times, times, times])
        out = resample(times, values, np.array([0.005, 0.2, 0.505, 1.0]))
        np.testing.assert_allclose(out[[0, 2], 0], [0.005, 0.505])
        self.assertTrue(np.isnan(out[[1, 3]]).all())

    def test_corner_stats_edges_and_spread(self):
        square = {0: [0, 0, 0], 1: [20, 0, 0], 2: [20, 20, 0], 3: [0, 20, 0]}
        visits = {corner: [(0, np.add(point, [0.1, 0, 0])), (1, np.add(point, [-0.1, 0, 0]))]
                  for corner, point in square.items()}
        stats, _means, edges = corner_stats(visits)
        np.testing.assert_allclose(edges, [20, 20, 20, 20])
        self.assertTrue(all(math.isclose(item["rms_mm"], 0.1) for item in stats.values()))

    def test_compare_to_commanded_scores_each_source_in_its_own_frame(self):
        corners = np.array([[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0]], dtype=float)
        t = np.arange(800) * 0.01
        corner_of = (np.arange(800) // 200) % 4
        cmd = corners[corner_of]
        in_dwell = np.ones(800, dtype=bool)
        windows = [(0, corner, t[200 * corner], t[200 * corner + 199]) for corner in range(4)]
        rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        tracker = (cmd + [5, 0, 0]) @ rotation.T + [100, 200, 300]  # rotated frame, x offset at every corner
        tracker[:10] = np.nan
        results, aligned = compare_to_commanded({"fk": cmd + [0, 0.5, 0], "mocap": tracker}, cmd, in_dwell,
                                                np.zeros(800, dtype=int), 1, [0, 1, 2], windows, t, corners)
        # A constant offset is absorbed by the rigid fit; only the frame change is reported.
        self.assertAlmostEqual(results["fk"]["dwell_rmse_mm"], 0.0, places=9)
        self.assertAlmostEqual(results["mocap"]["dwell_rmse_mm"], 0.0, places=6)
        self.assertAlmostEqual(results["mocap"]["fit_rotation_deg"], 90.0, places=6)
        self.assertAlmostEqual(results["mocap"]["coverage"], 790 / 800)
        self.assertTrue(math.isnan(results["mocap"]["moving_rmse_mm"]))
        np.testing.assert_allclose(aligned["mocap"][10:], cmd[10:], atol=1e-6)
        # A shape error (one corner displaced) survives the fit.
        bent = cmd.copy()
        bent[corner_of == 2] += [1.0, 0, 0]
        results, _ = compare_to_commanded({"fk": bent}, cmd, in_dwell, np.zeros(800, dtype=int), 1, [0, 1, 2],
                                          windows, t, corners)
        self.assertGreater(results["fk"]["dwell_rmse_mm"], 0.2)
        self.assertEqual(len(results["fk"]["corner_bias_mm"]), 4)


if __name__ == "__main__":
    unittest.main()
