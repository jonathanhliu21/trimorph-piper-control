import unittest

import numpy as np

from data_collection.collection import config
from data_collection.replay_actions import jacobian
from inference.executor.joint_control import DAMPING_SIGMA, JointController, damping

HOME = np.radians(config.PIPER_START_JOINTS_DEG['home_cube1'])
# Run 20261104_113450 at 101 s: J2/J3 oscillated here (smallest singular value 0.0145).
NEAR_SINGULAR = np.radians([-12.9, 162.6, -153.6, -50.8, 66.0, 47.4])


class JointControlTests(unittest.TestCase):
    def test_damping_only_below_threshold(self):
        self.assertEqual(damping(0.087), 0.0)
        self.assertEqual(damping(DAMPING_SIGMA), 0.0)
        self.assertGreater(damping(0.0145), 0.0)
        self.assertAlmostEqual(damping(0.0), DAMPING_SIGMA)

    def test_well_conditioned_target_is_solved_exactly(self):
        control = JointController()
        control.reset(HOME)
        target_joints = HOME + np.radians([1, -1, 1.5, 1, -1, 1])
        q, info = control.command(control.fk(target_joints), 1 / 15)
        self.assertEqual(info['damping'], 0.0)
        self.assertEqual(info['step_scale'], 1.0)
        self.assertLess(info['pos_err_mm'], 0.01)
        np.testing.assert_allclose(q, target_joints, atol=1e-4)

    def test_speed_cap_scales_the_whole_step(self):
        control = JointController((10, 10, 10, 10, 10, 10))
        control.reset(HOME)
        target_joints = HOME + np.radians([0, 5, 10, 0, 0, 0])
        q, info = control.command(control.fk(target_joints), 0.1)  # 1 deg allowed per joint
        step = np.degrees(q - HOME)
        self.assertAlmostEqual(np.max(np.abs(step)), 1.0, places=3)
        self.assertLess(info['step_scale'], 1.0)
        self.assertAlmostEqual(step[1] / step[2], 0.5, places=1)  # direction kept

    def test_damping_trades_pose_error_for_less_joint_motion_near_singularity(self):
        fk = JointController().fk
        self.assertLess(np.linalg.svd(jacobian(fk, NEAR_SINGULAR), compute_uv=False)[-1], DAMPING_SIGMA)
        target = fk(NEAR_SINGULAR + np.radians([0, 2, -4, 0, 0, 0]))
        moves = {}
        for sigma in (0.0, DAMPING_SIGMA):
            control = JointController((180,) * 6, sigma_start=sigma)
            control.reset(NEAR_SINGULAR)
            q, info = control.command(target, 1 / 15)
            moves[sigma] = (np.linalg.norm(q - NEAR_SINGULAR), info['pos_err_mm'])
        self.assertLess(moves[DAMPING_SIGMA][0], moves[0.0][0])
        self.assertGreater(moves[DAMPING_SIGMA][1], moves[0.0][1])

    def test_limits_and_wrist_side_are_respected(self):
        control = JointController((180,) * 6)
        control.reset(HOME)
        lower, upper = control.bounds()
        self.assertGreaterEqual(lower[4], np.radians(10))  # J5 stays positive
        unreachable = control.fk(HOME).copy()
        unreachable[:3] += [1.0, 0, 0]  # 1 m beyond the reach
        q, info = control.command(unreachable, 1 / 15)
        self.assertTrue(np.all(q >= control.lower - 1e-9) and np.all(q <= control.upper + 1e-9))
        self.assertGreater(info['pos_err_mm'], 100)


if __name__ == '__main__':
    unittest.main()
