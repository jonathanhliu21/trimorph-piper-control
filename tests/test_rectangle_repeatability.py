import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.calib.analyze_rectangle import corner_stats, fit_rigid, resample
from scripts.calib.rectangle_repeatability import build_schedule, parse_args, rectangle_corners, target_at


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


if __name__ == "__main__":
    unittest.main()
