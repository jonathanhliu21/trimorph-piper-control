"""Advisory margins must not disable physical bounds, failed IK or runtime gates."""
import unittest
import warnings
from dataclasses import replace
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
from data_collection import replay_actions as replay


class AdvisoryLimitTests(unittest.TestCase):
    def test_controller_bounds_retry_recovers_transient_invalid_response(self):
        robot = Mock()
        robot.get_joint_angle_vel_limits.side_effect = [
            SimpleNamespace(msg=SimpleNamespace(min_angle_limit=0., max_angle_limit=0.)),
            SimpleNamespace(msg=SimpleNamespace(min_angle_limit=-1., max_angle_limit=1.))]
        self.assertEqual(replay.read_controller_joint_limits(robot, 4), (-1., 1.))
        self.assertEqual(robot.get_joint_angle_vel_limits.call_count, 2)

    def test_controller_bounds_failure_identifies_joint_and_actual_values(self):
        robot = Mock()
        robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
            msg=SimpleNamespace(min_angle_limit=0., max_angle_limit=0.))
        with self.assertRaisesRegex(RuntimeError, r'J3.*min_angle_limit=0.0, max_angle_limit=0.0'):
            replay.read_controller_joint_limits(robot, 3)
        self.assertEqual(robot.get_joint_angle_vel_limits.call_count, 3)

    def setUp(self):
        self.q = np.array([.98, .2, -.3, .1, .5, .1])
        self.bounds = np.tile([-1., 1.], (6,1))
        self.fk = lambda q: np.asarray(q)
        self.limits = replay.Limits(warn_margins=True)

    def test_motion_mode_warns_for_source_steps_and_joint_speeds(self):
        seed = np.array([.1, .2, -.3, .1, .5, .1])
        poses = np.tile(np.r_[seed[:3], Rotation.from_euler('xyz', seed[3:]).as_rotvec()], (2,1))
        poses[1,0] += .4
        limits = replace(self.limits, warn_motion_limits=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', replay.AdvisoryLimitWarning)
            plan = replay.plan_actions(np.array([0., .01]), poses, np.zeros((2,13)), seed,
                                       self.bounds, self.fk, limits=limits)
        self.assertTrue(any('Cartesian position jump' in str(w.message) for w in caught))
        self.assertTrue(any('joint jump/speed' in str(w.message) for w in caught))
        self.assertAlmostEqual(plan['joints'][-1,0], .5, places=5)

    def test_motion_mode_warns_for_tracking_difference_and_continues(self):
        api = Mock()
        measured = self.q.copy(); measured[0] = 0.
        target = measured.copy(); target[0] = .4
        plan = dict(times=np.array([0.]), joints=target[None,:])
        with patch.object(replay, 'read_fresh_joints', return_value=measured), patch.object(replay, 'check_status'):
            with self.assertWarnsRegex(replay.AdvisoryLimitWarning, 'too far'):
                replay.play_plan(plan, api, None, self.bounds, self.fk,
                                 replace(self.limits, warn_motion_limits=True))
        api.move_joint.assert_called_once()

    def test_missed_deadline_shifts_remaining_schedule_instead_of_catchup(self):
        api = Mock()
        q = self.q.copy(); q[0] = 0.
        plan = dict(times=np.array([0., .05, .1]), joints=np.tile(q, (3,1)))
        with patch.object(replay.time, 'monotonic', side_effect=[0.,0.,.3,.3]), \
             patch.object(replay.time, 'sleep') as sleep, \
             patch.object(replay, 'read_fresh_joints', return_value=q), patch.object(replay, 'check_status'):
            with self.assertWarnsRegex(replay.AdvisoryLimitWarning, 'shifting schedule'):
                replay.play_plan(plan, api, None, self.bounds, self.fk,
                                 replace(self.limits, warn_motion_limits=True))
        self.assertEqual(api.move_joint.call_count, 3)
        self.assertAlmostEqual(sleep.call_args.args[0], .05)

    def test_margin_warns_but_physical_bound_still_raises(self):
        with self.assertWarnsRegex(replay.AdvisoryLimitWarning, 'Joint margin'):
            replay.check_joints(self.q, self.bounds, self.fk, self.limits)
        with self.assertRaisesRegex(ValueError, 'configured joint margin'):
            replay.check_joints(self.q, self.bounds, self.fk, replay.Limits())
        q = self.q.copy(); q[0] = 1.001
        with self.assertRaisesRegex(ValueError, 'Physical joint limit'):
            replay.check_joints(q, self.bounds, self.fk, self.limits)

    def test_wrist_and_jacobian_thresholds_warn(self):
        q = self.q.copy(); q[0] = 0.; q[4] = 0.
        with self.assertWarnsRegex(replay.AdvisoryLimitWarning, 'Wrist'):
            replay.check_joints(q, self.bounds, self.fk, self.limits)
        q[4] = .5
        with self.assertWarnsRegex(replay.AdvisoryLimitWarning, 'Jacobian'):
            replay.check_joints(q, self.bounds, lambda q: np.zeros(6), self.limits)

    def test_ik_can_use_margin_region_but_not_unreachable_target(self):
        poses = np.tile(np.r_[self.q[:3], Rotation.from_euler('xyz', self.q[3:]).as_rotvec()], (2,1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', replay.AdvisoryLimitWarning)
            plan = replay.plan_actions(np.array([0., 1.]), poses, np.zeros((2,13)), self.q,
                                       self.bounds, self.fk, limits=self.limits)
            self.assertTrue(np.all(plan['joints'] <= self.bounds[:,1]))
            poses[:,0] = 1.2
            with self.assertRaisesRegex(ValueError, 'no accurate IK'):
                replay.plan_actions(np.array([0., 1.]), poses, np.zeros((2,13)), self.q,
                                    self.bounds, self.fk, limits=self.limits)

    def test_large_runtime_tracking_error_still_stops_before_command(self):
        api = Mock()
        q = self.q.copy(); q[0] = 0.
        plan = dict(times=np.array([0.]), joints=(q+np.array([.4,0,0,0,0,0]))[None,:])
        with patch.object(replay, 'read_fresh_joints', return_value=q), patch.object(replay, 'check_status'):
            with self.assertRaisesRegex(RuntimeError, 'too far'):
                replay.play_plan(plan, api, None, self.bounds, self.fk, self.limits)
        api.move_joint.assert_not_called()
