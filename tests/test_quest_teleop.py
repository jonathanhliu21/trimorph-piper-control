import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.calib.get_coord_transform import transform_chunk
from scripts.teleop.quest_teleop import TeleopMapper, rate_limit

# Mirrored axes and a flange offset, like the 2026-10-31 Quest calibration.
A = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
CALIBRATION = {
    "linear_matrix": A.tolist(),
    "orientation": {"rotation_matrix": (-A).tolist()},
    "tracked_point_offset": {"vector": [-0.08, 0.02, -0.05], "frame": "piper_flange"},
}
HOME = [0.4, 0.01, -0.13, 3.13, 0.06, 1.48]


class QuestTeleopTests(unittest.TestCase):
    def test_position_delta_maps_through_calibration(self):
        mapper = TeleopMapper(CALIBRATION, HOME)
        q0, quat0 = np.array([0.1, 1.3, 0.5]), [0, 0, 0, 1]
        mapper.engage(q0, quat0, HOME[:3], Rotation.from_euler("xyz", HOME[3:]))
        position, rotation = mapper.target(q0 + [0.0, 0.0, -0.02], quat0)  # Quest -z -> Piper +x
        np.testing.assert_allclose(position - HOME[:3], [0.02, 0, 0], atol=1e-12)
        self.assertAlmostEqual((rotation * Rotation.from_euler("xyz", HOME[3:]).inv()).magnitude(), 0)

    def test_lever_compensation_is_opt_in_and_cancels_rotation_about_calibrated_point(self):
        home_r = Rotation.from_euler("xyz", HOME[3:])
        b, l = -A, np.array(CALIBRATION["tracked_point_offset"]["vector"])
        q0 = np.array([0.1, 1.3, 0.5])
        # Rotate the controller 40 deg about the calibrated (flange-equivalent) point.
        quest_delta = Rotation.from_euler("y", 40, degrees=True)
        robot_rotation = b @ quest_delta.as_matrix() @ b.T @ home_r.as_matrix()
        q = q0 + np.linalg.solve(A, (home_r.as_matrix() - robot_rotation) @ l)
        compensated = TeleopMapper(CALIBRATION, HOME, compensate_lever=True)
        compensated.engage(q0, [0, 0, 0, 1], HOME[:3], home_r)
        position, rotation = compensated.target(q, quest_delta.as_quat())
        np.testing.assert_allclose(position, HOME[:3], atol=1e-12)
        self.assertAlmostEqual((rotation * home_r.inv()).magnitude(), 0)  # position only: orientation held
        raw = TeleopMapper(CALIBRATION, HOME)  # default: raw tracked point
        raw.engage(q0, [0, 0, 0, 1], HOME[:3], home_r)
        self.assertGreater(np.linalg.norm(raw.target(q, quest_delta.as_quat())[0] - HOME[:3]), 0.03)

    def test_scale_and_offset_clamp(self):
        mapper = TeleopMapper(CALIBRATION, HOME, scale=2.0, max_offset_m=0.05)
        mapper.engage([0, 0, 0], [0, 0, 0, 1], HOME[:3], Rotation.from_euler("xyz", HOME[3:]))
        position, _ = mapper.target([0, 0.01, 0], [0, 0, 0, 1])
        np.testing.assert_allclose(position - HOME[:3], [0, 0.02, 0], atol=1e-12)
        position, _ = mapper.target([0, 0.2, 0], [0, 0, 0, 1])
        self.assertAlmostEqual(np.linalg.norm(position - HOME[:3]), 0.05)

    def test_orientation_mode_matches_transform_chunk(self):
        mapper = TeleopMapper(CALIBRATION, HOME, with_orientation=True, compensate_lever=True)
        robot_r = Rotation.from_euler("xyz", HOME[3:])
        q = np.array([[0.1, 1.3, 0.5], [0.11, 1.29, 0.52]])
        quats = Rotation.from_euler("xyz", [[10, 0, 0], [15, 5, -3]], degrees=True).as_quat()
        mapper.engage(q[0], quats[0], HOME[:3], robot_r)
        position, rotation = mapper.target(q[1], quats[1])
        expected_p, expected_q = transform_chunk(q, quats, HOME[:3], robot_r.as_quat(), CALIBRATION)
        np.testing.assert_allclose(position, expected_p[1], atol=1e-12)
        self.assertAlmostEqual((rotation * Rotation.from_quat(expected_q[1]).inv()).magnitude(), 0, places=9)

    def test_orientation_mode_rotates_about_flange_by_default(self):
        mapper = TeleopMapper(CALIBRATION, HOME, with_orientation=True)
        home_r = Rotation.from_euler("xyz", HOME[3:])
        mapper.engage([0.1, 1.3, 0.5], [0, 0, 0, 1], HOME[:3], home_r)
        quest_delta = Rotation.from_euler("y", 30, degrees=True)
        position, rotation = mapper.target([0.1, 1.3, 0.5], quest_delta.as_quat())  # rotate in place
        np.testing.assert_allclose(position, HOME[:3], atol=1e-12)                    # flange stays put
        expected = Rotation.from_matrix(-A @ quest_delta.as_matrix() @ -A.T) * home_r
        self.assertAlmostEqual((rotation * expected.inv()).magnitude(), 0, places=9)
        self.assertAlmostEqual(math.degrees((rotation * home_r.inv()).magnitude()), 30, places=6)

    def test_rotation_clamp(self):
        mapper = TeleopMapper(CALIBRATION, HOME, with_orientation=True, max_rotation_rad=math.radians(20))
        mapper.engage([0, 0, 0], [0, 0, 0, 1], HOME[:3], Rotation.from_euler("xyz", HOME[3:]))
        _, rotation = mapper.target([0, 0, 0], Rotation.from_euler("x", 60, degrees=True).as_quat())
        away = rotation * Rotation.from_euler("xyz", HOME[3:]).inv()
        self.assertAlmostEqual(math.degrees(away.magnitude()), 20, places=6)

    def test_rate_limit(self):
        start = (np.zeros(3), Rotation.identity())
        goal = (np.array([0.01, 0, 0]), Rotation.from_euler("z", 30, degrees=True))
        position, rotation = rate_limit(start, goal, 0.002, math.radians(5))
        np.testing.assert_allclose(position, [0.002, 0, 0])
        self.assertAlmostEqual(math.degrees(rotation.magnitude()), 5)
        position, rotation = rate_limit(start, goal, 1.0, 1.0)
        np.testing.assert_allclose(position, goal[0])


if __name__ == "__main__":
    unittest.main()
