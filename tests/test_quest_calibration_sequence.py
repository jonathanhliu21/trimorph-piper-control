import math
import unittest

import numpy as np

from scripts.calib.quest_calibration_sequence import (
    build_segments, check_joint_segments, circle_offset, envelope, parse_args, rectangle_offset,
)


class QuestCalibrationSequenceTests(unittest.TestCase):
    def test_shapes_start_and_end_at_home(self):
        rect, perimeter = rectangle_offset(0, 2, 0.03, 0.02)
        self.assertAlmostEqual(perimeter, 0.1)
        np.testing.assert_allclose(rect(0.0), 0, atol=1e-12)
        np.testing.assert_allclose(rect(0.3), [0.03, 0, 0], atol=1e-12)  # end of edge 0 at 30 % of perimeter
        np.testing.assert_allclose(rect(0.55), [0.025, 0, 0.02], atol=1e-12)  # middle of edge 2
        np.testing.assert_allclose(rect(0.9999), 0, atol=1e-5)
        circle, _ = circle_offset(1, 2, 0.015)
        np.testing.assert_allclose(circle(0.0), 0, atol=1e-12)
        np.testing.assert_allclose(circle(0.5), [0, 0.03, 0], atol=1e-12)

    def test_envelope_ramps_to_rest(self):
        self.assertEqual(envelope(0.0, 10, 2), 0.0)
        self.assertEqual(envelope(5.0, 10, 2), 1.0)
        self.assertAlmostEqual(envelope(1.0, 10, 2), 0.5)
        self.assertEqual(envelope(10.0, 10, 2), 0.0)

    def test_default_plan_is_safe_and_segments_return_home(self):
        args = parse_args([])
        segments = build_segments(args)
        self.assertEqual([s.label for s in segments][:3], ["warmup", "rect_xy", "circle_xy"])
        check_joint_segments(np.radians(args.home_joints_deg), segments, args.joint_margin_deg)
        for segment in segments:
            np.testing.assert_allclose(segment.offset(segment.duration_s), 0, atol=1e-9)

    def test_rejects_rotation_past_joint_limit(self):
        args = parse_args(["--rotation-deg", "20", "--rotation-joints", "6", "--dynamic-s", "0"])
        with self.assertRaisesRegex(ValueError, "rot_j6"):
            check_joint_segments(np.radians(args.home_joints_deg), build_segments(args), args.joint_margin_deg)

    def test_rejects_wrist_singularity_crossing(self):
        args = parse_args(["--rotation-deg", "30", "--rotation-joints", "5", "--dynamic-s", "0"])
        with self.assertRaisesRegex(ValueError, "singularity"):
            check_joint_segments(np.radians(args.home_joints_deg), build_segments(args), args.joint_margin_deg)

    def test_segment_filter(self):
        segments = build_segments(parse_args(["--segments", "dynamic", "rot_j4"]))
        self.assertEqual([s.label for s in segments], ["rot_j4", "dynamic"])
        with self.assertRaises(ValueError):
            build_segments(parse_args(["--segments", "nope"]))


if __name__ == "__main__":
    unittest.main()
