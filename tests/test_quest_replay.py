import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.teleop.quest_replay import (
    AbsoluteMapper, Trajectory, approach_path, calibration_origin, check_trajectory, quest_arrays, replay_clock,
    replay_wall_s,
)

# Mirrored axes, a flange offset and a per-recording origin, like the 2026-10-31 Quest calibration.
A = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
ORIGIN = np.array([0.96, -1.21, -0.14])
CALIBRATION = {
    "linear_matrix": A.tolist(),
    "orientation": {"rotation_matrix": (-A).tolist()},
    "tracked_point_offset": {"vector": [-0.04, 0.004, -0.067], "frame": "piper_flange"},
    "position_offset": None,
    "recordings": [{"position_offset": (ORIGIN + [0.001, 0, 0]).tolist()},
                   {"position_offset": (ORIGIN - [0.001, 0, 0]).tolist()}],
}
HOME = [0.4, 0.01, -0.13, 3.13, 0.06, 1.48]
LIMITS = dict(max_offset_m=0.15, max_rotation_rad=math.radians(30), max_speed_m_s=0.3,
              max_angular_speed_rad_s=math.radians(180), playback_speed=1.0, max_gap_s=0.2, quest_jump_m=0.05)


def quest_point_for(flange_position, flange_rotation):
    """Quest position that the calibration maps to this flange pose."""
    l = np.array(CALIBRATION["tracked_point_offset"]["vector"])
    return np.linalg.solve(A, flange_position - flange_rotation.apply(l) - ORIGIN)


def mapped(mapper, times, q, quat=None):
    quat = np.tile([0, 0, 0, 1.0], (len(q), 1)) if quat is None else quat
    return Trajectory(times, *mapper.map(q, quat))


class QuestReplayTests(unittest.TestCase):
    def test_origin_is_mean_of_recordings(self):
        origin, spread = calibration_origin(CALIBRATION)
        np.testing.assert_allclose(origin, ORIGIN)
        self.assertAlmostEqual(spread, 0.001)
        with self.assertRaises(ValueError):
            calibration_origin({"position_offset": None, "recordings": [{"position_offset": None}]})

    def test_absolute_mapping_inverts_calibration_equation(self):
        mapper = AbsoluteMapper(CALIBRATION, HOME)
        home_r = Rotation.from_euler("xyz", HOME[3:])
        goal = np.array([0.45, -0.02, -0.1])
        positions, rotations = mapper.map([quest_point_for(goal, home_r)], [[0, 0, 0, 1]])
        np.testing.assert_allclose(positions[0], goal, atol=1e-12)
        self.assertAlmostEqual((rotations[0] * home_r.inv()).magnitude(), 0)
        # Absolute, not anchored: the same Quest point maps to the same target wherever the recording starts.
        q = np.array([quest_point_for(goal + [0.05, 0, 0], home_r), quest_point_for(goal, home_r)])
        np.testing.assert_allclose(mapper.map(q, np.tile([0, 0, 0, 1.0], (2, 1)))[0][1], goal, atol=1e-12)

    def test_orientation_starts_at_home_and_follows_relative_rotation(self):
        mapper = AbsoluteMapper(CALIBRATION, HOME, with_orientation=True)
        home_r = Rotation.from_euler("xyz", HOME[3:])
        r0 = Rotation.from_euler("xyz", [20, -10, 5], degrees=True)
        delta = Rotation.from_euler("y", 25, degrees=True)
        q = np.zeros((2, 3))
        _, rotations = mapper.map(q, Rotation.concatenate([r0, delta * r0]).as_quat())
        self.assertAlmostEqual((rotations[0] * home_r.inv()).magnitude(), 0)
        expected = Rotation.from_matrix(-A @ delta.as_matrix() @ -A.T) * home_r
        self.assertAlmostEqual((rotations[1] * expected.inv()).magnitude(), 0, places=9)

    def test_replay_clock_eases_in_and_out(self):
        duration, speed, ease = 4.0, 0.5, 0.5
        wall = replay_wall_s(duration, speed, ease)
        self.assertAlmostEqual(wall, 8.5)
        grid = np.linspace(0, wall, 2001)
        clock = np.array([replay_clock(t, duration, speed, ease) for t in grid])
        self.assertEqual(clock[0], 0.0)
        self.assertAlmostEqual(clock[-1], duration)
        rate = np.diff(clock) / np.diff(grid)
        self.assertTrue(np.all(rate >= -1e-12) and np.all(rate <= speed + 1e-9))
        self.assertLess(rate[0], 0.01)   # starts at rest
        self.assertLess(rate[-1], 0.01)  # ends at rest
        self.assertAlmostEqual(replay_clock(2.0, duration, speed, 0.0), 1.0)

    def test_approach_is_smooth_and_ends_on_goal(self):
        start = (np.array([0.4, 0.0, -0.1]), Rotation.identity())
        goal = (np.array([0.45, 0.02, -0.12]), Rotation.from_euler("z", 20, degrees=True))
        path = list(approach_path(start, goal, 50.0, 0.05, math.radians(30)))
        np.testing.assert_allclose(path[-1][0], goal[0], atol=1e-12)
        self.assertAlmostEqual((path[-1][1] * goal[1].inv()).magnitude(), 0)
        positions = np.array([start[0]] + [p for p, _ in path])
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) * 50
        self.assertLessEqual(speeds.max(), 0.05 + 1e-9)
        self.assertLess(speeds[0], 0.002)
        self.assertLess(speeds[-1], 0.002)

    def test_check_accepts_slow_motion_near_home(self):
        mapper = AbsoluteMapper(CALIBRATION, HOME)
        home_r = Rotation.from_euler("xyz", HOME[3:])
        times = np.arange(0, 3, 1 / 72)
        flange = np.array(HOME[:3]) + np.outer(np.sin(times), [0.03, 0, 0])
        q = np.array([quest_point_for(p, home_r) for p in flange])
        problems, stats = check_trajectory(mapped(mapper, times, q), times, q, mapper, **LIMITS)
        self.assertEqual(problems, [])
        self.assertLess(stats["max_offset_mm"], 31)

    def test_check_rejects_far_fast_and_jumping_recordings(self):
        mapper = AbsoluteMapper(CALIBRATION, HOME)
        home_r = Rotation.from_euler("xyz", HOME[3:])
        times = np.arange(0, 2, 1 / 72)
        flange = np.array(HOME[:3]) + np.outer(times, [0.4, 0, 0])  # 400 mm/s, ends 800 mm away
        flange[100:] += [0, 0, 0.06]                                  # tracking jump
        q = np.array([quest_point_for(p, home_r) for p in flange])
        problems, _ = check_trajectory(mapped(mapper, times, q), times, q, mapper, **LIMITS)
        text = " ".join(problems)
        for expected in ("from home", "peak speed", "jump"):
            self.assertIn(expected, text)
        smooth = np.array([quest_point_for(p, home_r) for p in np.array(HOME[:3]) + np.outer(times, [0.4, 0, 0])])
        slow = dict(LIMITS, playback_speed=0.5)  # 200 mm/s on the arm
        problems, _ = check_trajectory(mapped(mapper, times, smooth), times, smooth, mapper, **slow)
        self.assertNotIn("peak speed", " ".join(problems))

    def test_quest_arrays_drop_invalid_and_duplicate_samples(self):
        rows = [{"monotonic_ns": ns, "px": x, "py": 0, "pz": 0, "qx": 0, "qy": 0, "qz": 0, "qw": w}
                for ns, x, w in ((1_000_000_000, 0.1, 2.0), (1_010_000_000, 0.2, 0.0),
                                 (1_020_000_000, 0.3, 1.0), (1_020_000_000, 0.3, 1.0))]
        times, q, quat = quest_arrays(rows)
        np.testing.assert_allclose(times, [0.0, 0.02])
        np.testing.assert_allclose(q[:, 0], [0.1, 0.3])
        np.testing.assert_allclose(quat[0], [0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
