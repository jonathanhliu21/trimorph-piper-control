"""Offline planning and mocked execution; never connects to hardware."""
import contextlib
from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from data_collection import replay_actions as replay


@dataclass
class HandConfigFixture:
    operating_mode: int = 5

    @property
    def joint_lower_limits_array(self):
        return np.full(13, -3.)

    @property
    def joint_upper_limits_array(self):
        return np.full(13, 3.)


class ActionReplayTests(unittest.TestCase):
    def setUp(self):
        self.seed = np.array([.1, .2, .3, .1, .4, .1])
        self.bounds = np.tile([-3., 3.], (6, 1))
        self.fk = lambda q: np.array(q)
        original_savez = np.savez
        def isolated_savez(file, *args, **kwargs):
            if isinstance(file,(str,Path)) and Path(file).resolve().parent == Path('/tmp'):
                raise AssertionError('Replay tests must write plans inside their TemporaryDirectory')
            return original_savez(file,*args,**kwargs)
        guard = patch.object(replay.np,'savez',side_effect=isolated_savez)
        guard.start()
        self.addCleanup(guard.stop)

    def actions(self):
        pose = np.r_[self.seed[:3], Rotation.from_euler('xyz', self.seed[3:]).as_rotvec()]
        poses = np.tile(pose, (2, 1))
        poses[1, 0] += .001
        return np.array([0., .2]), poses, np.zeros((2, 13))

    def plan(self, *values, **kwargs):
        return replay.plan_actions(*(values or self.actions()), self.seed, self.bounds, self.fk, **kwargs)

    def test_preposition_adds_checked_slow_approach_before_absolute_trajectory(self):
        times, poses, hand = self.actions()
        poses[:, 0] += .5
        with self.assertRaisesRegex(ValueError, 'Frame 0: joint jump'):
            self.plan(times, poses, hand)
        trajectory = self.plan(times, poses, hand, preposition=True)
        plan = replay.prepend_preposition(trajectory, self.seed, self.bounds, self.fk, replay.Limits())
        np.testing.assert_allclose(plan['joints'][0], self.seed)
        np.testing.assert_allclose(plan['poses'][-len(trajectory['poses']):], trajectory['poses'])
        offset = len(plan['times']) - len(trajectory['times'])
        self.assertAlmostEqual(plan['times'][offset], np.degrees(.5) / 5.)
        speed = np.abs(np.diff(plan['joints'][:offset+1], axis=0)) / np.diff(plan['times'][:offset+1])[:, None]
        self.assertLessEqual(np.degrees(speed).max(), 5. + 1e-8)
        self.assertTrue(np.all(np.diff(plan['times']) > 0))

    def test_preposition_still_rejects_wrist_singularity_crossing(self):
        trajectory = self.plan()
        trajectory['joints'][:, 4] = -.4
        with self.assertRaisesRegex(ValueError, 'Wrist singularity'):
            replay.prepend_preposition(trajectory, self.seed, self.bounds, self.fk, replay.Limits())

    def test_preposition_planning_remains_offline_with_explicit_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, poses, hand = self.actions()
            poses[:, 0] += .5
            np.save(root/'actions.npy', poses)
            with patch.object(replay, 'piper_model', return_value=(self.fk, self.bounds.copy())), \
                    patch('scripts.piper_api.piper_api.PiperAPI') as api, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = replay.main([str(root/'actions.npy'), '--hz', '5', '--pose-frame', 'piper_base',
                                      '--arm-only', '--preposition', '--start-joints-deg',
                                      *map(str, np.degrees(self.seed)), '--output', str(root/'plan.npz')])
            self.assertEqual(result, 0)
            api.assert_not_called()
            with np.load(root/'plan.npz') as saved:
                np.testing.assert_allclose(saved['joints'][0], self.seed)
                self.assertEqual(saved['hand'].shape[1], 0)

    def test_loader_aligns_dataset_hand_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['absolute_action_t', 'processed_encoder_t', 'syncs']:
                (root/name).mkdir()
            path = root/'absolute_action_t/000001.npy'
            np.save(path, np.zeros((2, 6)))
            np.save(root/'processed_encoder_t/000001.npy', np.tile(np.array([0., .2, .4])[:, None], (1, 13)))
            np.savez(root/'syncs/000001.npz', pose_frame='piper_base',
                     pose_monotonic_ns=[1_100_000_000, 1_300_000_000],
                     encoder_monotonic_ns=[1_000_000_000, 1_200_000_000, 1_400_000_000])
            times, poses, hand = replay.load_actions(path)
            np.testing.assert_allclose(times, [0., .2])
            np.testing.assert_allclose(hand[:, 0], [.1, .3])

    def test_relative_frame_and_extrapolation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'actions.npy'
            np.save(path, np.zeros((2, 6)))
            np.save(root/'hand.npy', np.zeros((2, 13)))
            for frame, stamps, message in [
                ('relative_to_episode_start', [0, 1], 'piper_base'),
                ('piper_base', [2, 3], 'extrapolation'),
            ]:
                np.savez(root/'sync.npz', pose_frame=frame, pose_monotonic_ns=stamps,
                         encoder_monotonic_ns=[0, 1])
                with self.assertRaisesRegex(ValueError, message):
                    replay.load_actions(path, hand_path=root/'hand.npy', sync_path=root/'sync.npz')

    def test_standalone_combined_requires_explicit_timing_and_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'actions.npy'
            np.save(path, np.zeros((2, 19)))
            with self.assertRaises(ValueError):
                replay.load_actions(path)
            times, poses, hand = replay.load_actions(path, hz=15, pose_frame='piper_base')
            self.assertEqual(hand.shape, (2, 13))
            np.testing.assert_allclose(times, [0, 1/15])

    def test_plan_is_continuous_and_slowed(self):
        plan = self.plan(rate_hz=20, playback_speed=.2)
        self.assertAlmostEqual(plan['times'][-1], 1.)
        self.assertEqual(plan['joints'].shape, (21, 6))
        np.testing.assert_allclose(plan['joints'][0], self.seed, atol=1e-7)
        self.assertTrue(np.all(plan['jacobian_sigma'] > .005))

    def test_one_sample_plan(self):
        times, poses, hand = self.actions()
        plan = self.plan(times[:1], poses[:1], hand[:1])
        self.assertEqual(len(plan['joints']), 1)

    def test_bad_source_jumps_are_rejected_before_resampling(self):
        for kind, message in [('position', 'position jump'), ('rotation', 'rotation jump'), ('hand', 'hand position jump')]:
            times, poses, hand = self.actions()
            if kind == 'position':
                poses[1, 0] += .2
            elif kind == 'rotation':
                poses[1, 3:] = Rotation.from_euler('xyz', [1., .4, .1]).as_rotvec()
            else:
                hand[1, 0] = .2
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, message):
                self.plan(times, poses, hand)

    def test_joint_limit_and_wrist_margin_rejected(self):
        for joint, angle, message in [(0, 3., 'Joint limit'), (4, 0., 'Wrist singularity')]:
            q = self.seed.copy()
            q[joint] = angle
            with self.assertRaisesRegex(ValueError, message):
                replay.check_joints(q, self.bounds, self.fk, replay.Limits())

    def test_joint_error_reports_joint_angle_bounds_and_margin(self):
        q = self.seed.copy()
        q[3] = np.radians(102.837)
        bounds = self.bounds.copy()
        bounds[3] = np.radians([-100.,100.])
        with self.assertRaises(ValueError) as error:
            replay.check_joints(q,bounds,self.fk,replay.Limits())
        message = str(error.exception)
        self.assertIn('J4=102.837', message)
        self.assertIn('[-97.000, 97.000]', message)
        self.assertIn('margin 3 deg', message)

    def test_check_start_reads_limits_but_never_enables_or_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, poses, fingers = self.actions()
            np.save(root/'actions.npy',np.c_[poses,fingers])
            api = Mock()
            api.robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
                msg=SimpleNamespace(min_angle_limit=-3.,max_angle_limit=3.))
            with patch.object(replay,'piper_model',return_value=(self.fk,self.bounds.copy())), \
                 patch('scripts.piper_api.piper_api.PiperAPI',return_value=api), \
                 patch.object(replay,'read_fresh_joints',return_value=self.seed), \
                 patch.object(replay,'check_status'), patch.object(replay,'plan_actions') as plan, \
                 patch('inference.executor.executor.load_hand_class') as hand, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                result = replay.main([str(root/'actions.npy'),'--hz','5','--pose-frame','piper_base','--check-start',
                                      '--output',str(root/'plan.npz')])
            self.assertEqual(result,0)
            self.assertIn('controller=',output.getvalue())
            self.assertIn('planning=',output.getvalue())
            api.enable.assert_not_called()
            api.move_joint.assert_not_called()
            plan.assert_not_called()
            hand.assert_not_called()
            api.disconnect.assert_called_once()

    def test_jacobian_singularity_rejected(self):
        def singular(q):
            pose = q.copy()
            pose[0] = 0
            return pose
        with self.assertRaisesRegex(ValueError, 'Jacobian'):
            replay.check_joints(self.seed, self.bounds, singular, replay.Limits())

    def test_wrist_zero_crossing_rejected_between_valid_endpoints(self):
        seed = self.seed.copy()
        seed[4] = .2
        goal = seed.copy()
        goal[4] = -.2
        poses = np.array([np.r_[q[:3], Rotation.from_euler('xyz', q[3:]).as_rotvec()]
                          for q in (seed, goal)])
        limits = replay.Limits(max_joint_step_deg=30, max_joint_speed_deg_s=60,
                               max_rotation_step_deg=30)
        with self.assertRaisesRegex(ValueError, 'Wrist singularity'):
            replay.plan_actions(np.array([0., 1.]), poses, np.zeros((2,13)), seed,
                                self.bounds, self.fk, rate_hz=1, playback_speed=1, limits=limits)

    def test_loader_trims_common_coverage_without_extrapolating(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root/'actions.npy', np.zeros((4,6)))
            np.save(root/'hand.npy', np.zeros((2,13)))
            np.savez(root/'sync.npz', pose_frame='piper_base',
                     pose_monotonic_ns=[0, 1_000_000_000, 2_000_000_000, 3_000_000_000],
                     encoder_monotonic_ns=[1_000_000_000, 2_000_000_000])
            with contextlib.redirect_stdout(io.StringIO()):
                times, poses, hand = replay.load_actions(root/'actions.npy', hand_path=root/'hand.npy', sync_path=root/'sync.npz')
            np.testing.assert_allclose(times, [0, 1])
            self.assertEqual(poses.shape, (2,6))
            self.assertEqual(hand.shape, (2,13))

    def test_large_initial_joint_gap_rejected(self):
        times, poses, hand = self.actions()
        poses[:, 0] += .1
        with self.assertRaisesRegex(ValueError, 'joint jump/speed'):
            self.plan(times, poses, hand)

    def test_unreachable_target_rejected(self):
        times, poses, hand = self.actions()
        poses[:, 0] = 10
        with self.assertRaisesRegex(ValueError, 'no accurate IK'):
            self.plan(times, poses, hand)

    def test_hand_speed_error_identifies_motor_and_velocity(self):
        times, poses, hand = self.actions()
        times = np.array([0., .01])
        hand[1,4] = .09
        with self.assertRaisesRegex(ValueError, r'motor 4, .*rad/s.*playback speed'):
            self.plan(times,poses,hand)

    def test_stale_feedback_rejected(self):
        api = Mock()
        api.get_joint_state.return_value = SimpleNamespace(timestamp=time.time()-1, joints=self.seed)
        with self.assertRaisesRegex(RuntimeError, '250 ms'):
            replay.read_fresh_joints(api)

    def test_post_planning_feedback_wait_recovers_without_relaxing_age_limit(self):
        api = Mock()
        api.get_joint_state.side_effect = [
            SimpleNamespace(timestamp=99., joints=self.seed),
            SimpleNamespace(timestamp=100., joints=self.seed)]
        with patch.object(replay.time, 'time', return_value=100.), \
                patch.object(replay.time, 'sleep') as sleep, \
                patch.object(replay, 'check_status'), contextlib.redirect_stdout(io.StringIO()):
            np.testing.assert_array_equal(replay.wait_for_fresh_feedback(api), self.seed)
        sleep.assert_called_once()
        api.enable.assert_not_called()

    def test_post_planning_feedback_wait_times_out_before_enable(self):
        api = Mock()
        api.get_joint_state.return_value = SimpleNamespace(timestamp=99., joints=self.seed)
        with patch.object(replay.time, 'time', return_value=100.), \
                self.assertRaisesRegex(RuntimeError, 'arm was not enabled'):
            replay.wait_for_fresh_feedback(api, timeout_s=0.)
        api.enable.assert_not_called()

    def test_custom_joint_speed_cli_controls_planning_and_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, poses, _ = self.actions()
            poses[1, 0] = poses[0, 0] + .1  # ~29 deg/s with this test FK.
            np.save(root/'actions.npy', poses)
            options = [str(root/'actions.npy'), '--arm-only', '--hz', '5',
                       '--playback-speed', '1', '--pose-frame', 'piper_base',
                       '--start-joints-deg', *map(str, np.degrees(self.seed)),
                       '--output', str(root/'plan.npz')]
            with patch.object(replay, 'piper_model', return_value=(self.fk, self.bounds.copy())), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'joint jump/speed'):
                    replay.main(options)
                replay.main([*options, '--max-joint-speed-deg-s', '60'])
            with np.load(root/'plan.npz') as plan:
                self.assertEqual(json.loads(str(plan['limits_json']))['max_joint_speed_deg_s'], 60.)
        for invalid in ['0', '-1', 'nan', 'inf']:
            with self.subTest(invalid=invalid), patch('sys.stderr'), self.assertRaises(SystemExit):
                replay.parse_args(['actions.npy', '--max-joint-speed-deg-s', invalid])

    def test_feedback_failure_diagnostics_distinguish_causes(self):
        api = Mock()
        for stamp, reason in [(99., 'stale feedback'), (101., 'future timestamp'),
                              (float('nan'), 'nonfinite timestamp')]:
            with self.subTest(reason=reason), patch.object(replay.time, 'time', return_value=100.):
                api.get_joint_state.return_value = SimpleNamespace(
                    timestamp=stamp, joints=self.seed, hz=200.)
                with self.assertRaisesRegex(RuntimeError, f'reason={reason}.*sdk_hz=200'):
                    replay.read_fresh_joints(api)
        api.get_joint_state.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'reason=missing feedback'):
            replay.read_fresh_joints(api)

    def test_feedback_diagnostics_preserve_freshness_boundary(self):
        api = Mock()
        with patch.object(replay.time, 'time', return_value=100.):
            api.get_joint_state.return_value = SimpleNamespace(timestamp=99.75, joints=self.seed)
            np.testing.assert_array_equal(replay.read_fresh_joints(api), self.seed)
            api.get_joint_state.return_value.timestamp = 99.749
            with self.assertRaisesRegex(RuntimeError, 'age_ms=251.000'):
                replay.read_fresh_joints(api)

    def test_execute_validates_before_enabling_and_uses_smooth_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, poses, fingers = self.actions()
            np.save(root/'actions.npy', np.c_[poses, fingers])
            api, hand = Mock(), Mock()
            hand.read_pos.return_value = np.zeros(13)
            api.robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
                msg=SimpleNamespace(min_angle_limit=-3., max_angle_limit=3.))
            with patch.object(replay, 'piper_model', return_value=(self.fk, self.bounds.copy())), \
                 patch('scripts.piper_api.piper_api.PiperAPI', return_value=api), \
                 patch.object(replay, 'read_fresh_joints', return_value=self.seed), \
                 patch.object(replay, 'check_status'), \
                 patch('inference.executor.executor.load_hand_class', return_value=Mock(return_value=hand)), \
                 patch('data_collection.collection.hand_common.load_hand_config', return_value=HandConfigFixture()), \
                 patch('data_collection.collection.hand_common.verify_connected_hand'), \
                 patch.object(replay, 'play_plan') as play, \
                 contextlib.redirect_stdout(io.StringIO()):
                replay.main([str(root/'actions.npy'), '--hz', '5', '--pose-frame', 'piper_base',
                             '--execute', '--output', str(root/'plan.npz')])
            self.assertEqual(api.robot.get_joint_angle_vel_limits.call_count, 6)
            api.enable.assert_called_once()
            api.set_speed_percent.assert_called_once_with(10)
            play.assert_called_once()
            hand.close.assert_called_once_with(disable_torque=False)
            api.move_eef.assert_not_called()

    def test_failed_preflight_never_enables_actuators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, poses, fingers = self.actions()
            poses[:, 0] += .1
            np.save(root/'actions.npy', np.c_[poses, fingers])
            api = Mock()
            api.robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
                msg=SimpleNamespace(min_angle_limit=-3., max_angle_limit=3.))
            with patch.object(replay, 'piper_model', return_value=(self.fk, self.bounds.copy())), \
                 patch('scripts.piper_api.piper_api.PiperAPI', return_value=api), \
                 patch.object(replay, 'read_fresh_joints', return_value=self.seed), \
                 patch.object(replay, 'check_status'), \
                 patch('inference.executor.executor.load_hand_class') as hand_factory, \
                 self.assertRaisesRegex(ValueError, 'joint jump/speed'):
                replay.main([str(root/'actions.npy'), '--hz', '5', '--pose-frame', 'piper_base', '--execute',
                             '--output',str(root/'plan.npz')])
            api.enable.assert_not_called()
            hand_factory.assert_not_called()
            api.robot.electronic_emergency_stop.assert_not_called()

    def test_execution_fault_requests_controller_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, poses, fingers = self.actions()
            np.save(root/'actions.npy', np.c_[poses, fingers])
            api, hand = Mock(), Mock()
            hand.read_pos.return_value = np.zeros(13)
            api.robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
                msg=SimpleNamespace(min_angle_limit=-3., max_angle_limit=3.))
            with patch.object(replay, 'piper_model', return_value=(self.fk, self.bounds.copy())), \
                 patch('scripts.piper_api.piper_api.PiperAPI', return_value=api), \
                 patch.object(replay, 'read_fresh_joints', return_value=self.seed), \
                 patch.object(replay, 'check_status'), \
                 patch('inference.executor.executor.load_hand_class', return_value=Mock(return_value=hand)), \
                 patch('data_collection.collection.hand_common.load_hand_config', return_value=HandConfigFixture()), \
                 patch('data_collection.collection.hand_common.verify_connected_hand'), \
                 patch.object(replay, 'play_plan', side_effect=RuntimeError('tracking fault')), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'tracking fault'):
                replay.main([str(root/'actions.npy'), '--hz', '5', '--pose-frame', 'piper_base', '--execute',
                             '--output',str(root/'plan.npz')])
            api.robot.electronic_emergency_stop.assert_called_once()
            hand.close.assert_called_once_with(disable_torque=False)

    def test_play_uses_smooth_joints_and_rejects_tracking_error(self):
        plan = self.plan(rate_hz=5)
        api, hand = Mock(), Mock()
        with patch.object(replay, 'read_fresh_joints', return_value=self.seed), \
             patch.object(replay, 'check_status'), patch.object(replay.time, 'sleep'):
            replay.play_plan(plan, api, hand, self.bounds, self.fk, replay.Limits())
        self.assertEqual(api.move_joint.call_count, len(plan['times']))
        self.assertTrue(all(c.kwargs == {'mode': 'smooth'} for c in api.move_joint.call_args_list))
        api.move_eef.assert_not_called()
        api.reset_mock()
        hand.reset_mock()
        with patch.object(replay, 'read_fresh_joints', return_value=self.seed+np.array([.1,0,0,0,0,0])), \
             patch.object(replay, 'check_status'), self.assertRaisesRegex(RuntimeError, 'too far'):
            replay.play_plan(plan, api, hand, self.bounds, self.fk, replay.Limits())
        api.move_joint.assert_not_called()
        hand.set_positions.assert_not_called()

    def test_default_main_is_offline_and_exports_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, poses, hand = self.actions()
            np.save(root/'actions.npy', np.c_[poses, hand])
            with patch.object(replay, 'piper_model', return_value=(self.fk, self.bounds)), \
                 patch('scripts.piper_api.piper_api.PiperAPI') as api, \
                 contextlib.redirect_stdout(io.StringIO()):
                replay.main([str(root/'actions.npy'), '--hz', '5', '--pose-frame', 'piper_base',
                             '--start-joints-deg', *map(str, np.degrees(self.seed)),
                             '--output', str(root/'plan.npz')])
            api.assert_not_called()
            with np.load(root/'plan.npz', allow_pickle=False) as saved:
                self.assertFalse(saved['controller_limits_verified'])
                self.assertEqual(saved['joints'].shape[1], 6)

    def test_sync_npz_cannot_be_mistaken_for_joint_angles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sync.npz'
            np.savez(path, start_pose=self.seed, pose_frame='piper_base')
            with self.assertRaisesRegex(ValueError, 'not six joint angles'):
                replay.load_joint_plan(path)

    def test_saved_joint_plan_defaults_to_exact_absolute_joints(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root/'saved.npz', **plan)
            with patch.object(replay, 'piper_model', return_value=(self.fk, self.bounds)), \
                 patch.object(replay, 'least_squares', side_effect=AssertionError('No IK for joint plans')), \
                 patch('scripts.piper_api.piper_api.PiperAPI') as api, \
                 contextlib.redirect_stdout(io.StringIO()):
                replay.main([str(root/'saved.npz'), '--output', str(root/'validated.npz')])
            api.assert_not_called()
            with np.load(root/'validated.npz', allow_pickle=False) as result:
                np.testing.assert_array_equal(result['joints'], plan['joints'])
                np.testing.assert_array_equal(result['times'], plan['times'])
                np.testing.assert_array_equal(result['hand'], plan['hand'])
                self.assertEqual(str(result['replay_mode']), 'absolute')

    def test_relative_joint_plan_reanchors_and_rechecks_limits(self):
        plan = self.plan()
        shifted = self.seed + np.array([.1,0,0,0,0,0])
        result = replay.validate_joint_plan(plan['times'], plan['joints'], plan['hand'],
            shifted, self.bounds, self.fk, relative=True)
        np.testing.assert_allclose(result['joints'][0], shifted)
        np.testing.assert_allclose(np.diff(result['joints'],axis=0), np.diff(plan['joints'],axis=0))
        np.testing.assert_array_equal(result['hand'], plan['hand'])
        with self.assertRaisesRegex(ValueError, 'initial target'):
            replay.validate_joint_plan(plan['times'], plan['joints'], plan['hand'],
                shifted, self.bounds, self.fk)
        shifted[0] = 2.948
        with self.assertRaisesRegex(ValueError, 'Joint limit'):
            replay.validate_joint_plan(plan['times'], plan['joints'], plan['hand'],
                shifted, self.bounds, self.fk, relative=True)

    def test_relative_cartesian_poses_preserve_spatial_deltas(self):
        _, poses, _ = self.actions()
        start = np.array([.6, -.2, .2, -.2, .4, .7])
        result = replay.anchor_poses(poses, start)
        np.testing.assert_allclose(result[0,:3], start[:3])
        np.testing.assert_allclose(np.diff(result[:,:3],axis=0), np.diff(poses[:,:3],axis=0))
        np.testing.assert_allclose(Rotation.from_rotvec(result[0,3:]).as_matrix(),
                                   Rotation.from_euler('xyz',start[3:]).as_matrix(),atol=1e-12)

    def test_saved_joint_plan_does_not_trust_old_safety_checks(self):
        plan = self.plan()
        plan['joints'][-1,0] += .2
        with self.assertRaisesRegex(ValueError, 'joint jump'):
            replay.validate_joint_plan(plan['times'],plan['joints'],plan['hand'],
                                      self.seed,self.bounds,self.fk)

    def test_relative_mode_defaults_to_observed_start(self):
        args = replay.parse_args(['saved.npz','--relative'])
        self.assertIsNone(args.start_joints_deg)
        self.assertFalse(args.execute)

    def test_planning_reads_current_start_without_enabling_actuators(self):
        for kind, relative in [('npy',True), ('npz',True), ('npy',False)]:
            with self.subTest(kind=kind,relative=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, poses, fingers = self.actions()
                if relative:
                    poses[:,:3] += [1.,2.,3.]
                if kind == 'npy':
                    source = root/'actions.npy'
                    np.save(source,np.c_[poses,fingers])
                    options = ['--hz','5','--pose-frame','piper_base']
                else:
                    source = root/'actions.npz'
                    plan = self.plan()
                    plan['joints'] += np.array([.1,0,0,0,0,0])
                    np.savez(source,**plan)
                    options = []
                api = Mock()
                api.robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
                    msg=SimpleNamespace(min_angle_limit=-3.,max_angle_limit=3.))
                output = io.StringIO()
                with patch.object(replay,'piper_model',return_value=(self.fk,self.bounds.copy())), \
                     patch('scripts.piper_api.piper_api.PiperAPI',return_value=api) as factory, \
                     patch.object(replay,'read_fresh_joints',return_value=self.seed) as read, \
                     patch.object(replay,'check_status'), \
                     patch('inference.executor.executor.load_hand_class') as hand, \
                     patch.object(replay,'play_plan') as play, contextlib.redirect_stdout(output):
                    replay.main([str(source),*options,*(['--relative'] if relative else []),
                                 '--output',str(root/'plan.npz')])
                factory.assert_called_once_with(firmware='v183',channel='can0',installation_pos=None,payload=None)
                read.assert_called_once_with(api,check_freshness=True)
                self.assertEqual(api.robot.get_joint_angle_vel_limits.call_count,6)
                api.enable.assert_not_called()
                api.set_speed_percent.assert_not_called()
                api.move_joint.assert_not_called()
                api.move_eef.assert_not_called()
                api.robot.electronic_emergency_stop.assert_not_called()
                hand.assert_not_called()
                play.assert_not_called()
                api.disconnect.assert_called_once()
                with np.load(root/'plan.npz',allow_pickle=False) as result:
                    np.testing.assert_allclose(result['seed_rad'],self.seed)
                    np.testing.assert_allclose(result['joints'][0],self.seed,atol=1e-7)
                    self.assertTrue(result['controller_limits_verified'])
                self.assertIn('no actuators enabled or targets sent',output.getvalue())

    def test_read_only_planning_failure_disconnects_without_enabling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _,poses,fingers = self.actions()
            np.save(root/'actions.npy',np.c_[poses,fingers])
            api = Mock()
            api.robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
                msg=SimpleNamespace(min_angle_limit=-3.,max_angle_limit=3.))
            with patch.object(replay,'piper_model',return_value=(self.fk,self.bounds.copy())), \
                 patch('scripts.piper_api.piper_api.PiperAPI',return_value=api), \
                 patch.object(replay,'read_fresh_joints',side_effect=RuntimeError('stale joints')), \
                 patch('inference.executor.executor.load_hand_class') as hand, \
                 self.assertRaisesRegex(RuntimeError,'stale joints'):
                replay.main([str(root/'actions.npy'),'--relative','--hz','5','--pose-frame','piper_base',
                             '--output',str(root/'plan.npz')])
            api.enable.assert_not_called()
            hand.assert_not_called()
            api.robot.electronic_emergency_stop.assert_not_called()
            api.disconnect.assert_called_once()

    def test_arm_only_executes_without_hand_loading_or_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, poses, fingers = self.actions()
            fingers[:] = np.nan  # Hand data must not block arm-only diagnostics.
            np.save(root/'actions.npy',np.c_[poses,fingers])
            api = Mock()
            api.robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
                msg=SimpleNamespace(min_angle_limit=-3.,max_angle_limit=3.))
            with patch.object(replay,'piper_model',return_value=(self.fk,self.bounds.copy())), \
                 patch('scripts.piper_api.piper_api.PiperAPI',return_value=api), \
                 patch.object(replay,'read_fresh_joints',return_value=self.seed), \
                 patch.object(replay,'check_status'), \
                 patch('inference.executor.executor.load_hand_class') as hand, \
                 patch.object(replay.time,'sleep'), contextlib.redirect_stdout(io.StringIO()):
                replay.main([str(root/'actions.npy'),'--arm-only','--execute','--hz','5',
                             '--pose-frame','piper_base','--output',str(root/'plan.npz')])
            hand.assert_not_called()
            api.enable.assert_called_once()
            self.assertGreater(api.move_joint.call_count,0)
            api.move_eef.assert_not_called()
            with np.load(root/'plan.npz',allow_pickle=False) as plan:
                self.assertEqual(plan['hand'].shape[1],0)
                self.assertEqual(str(plan['devices']),'arm')
            with self.assertRaises(ValueError):
                replay.load_hand_actions(root/'plan.npz')

    def test_arm_only_dataset_does_not_need_hand_files_or_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'absolute_action_t').mkdir()
            (root/'syncs').mkdir()
            _,poses,_=self.actions()
            path=root/'absolute_action_t/000000.npy'
            np.save(path,poses)
            np.savez(root/'syncs/000000.npz',pose_frame='piper_base',pose_monotonic_ns=[0,200000000])
            times,arm,hand=replay.load_actions(path,arm_only=True)
            np.testing.assert_allclose(arm,poses)
            np.testing.assert_allclose(times,[0,.2])

    def test_hand_only_executes_without_piper_or_ik(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,poses,fingers=self.actions()
            poses[:]=np.nan  # Arm data must not block hand-only diagnostics.
            np.save(root/'actions.npy',np.c_[poses,fingers])
            hand=Mock()
            hand.read_pos.return_value=np.zeros(13)
            with patch('scripts.piper_api.piper_api.PiperAPI') as api, \
                 patch.object(replay,'piper_model') as model, \
                 patch.object(replay,'least_squares') as solve, \
                 patch('inference.executor.executor.load_hand_class',return_value=Mock(return_value=hand)), \
                 patch('data_collection.collection.hand_common.load_hand_config',return_value=HandConfigFixture()), \
                 patch('data_collection.collection.hand_common.verify_connected_hand'), \
                 patch.object(replay.time,'sleep'), contextlib.redirect_stdout(io.StringIO()):
                replay.main([str(root/'actions.npy'),'--hand-only','--execute','--hz','5',
                             '--output',str(root/'plan.npz')])
            api.assert_not_called()
            model.assert_not_called()
            solve.assert_not_called()
            hand.enable_torque.assert_called_once()
            self.assertGreater(hand.set_positions.call_count,1)
            hand.close.assert_called_once_with(disable_torque=False)
            with np.load(root/'plan.npz',allow_pickle=False) as plan:
                self.assertEqual(plan['joints'].shape[1],0)
                self.assertEqual(str(plan['devices']),'hand')
            with self.assertRaises(ValueError):
                replay.load_joint_plan(root/'plan.npz',arm_only=True)

    def test_hand_only_dataset_needs_only_hand_and_encoder_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'processed_encoder_t').mkdir()
            (root/'syncs').mkdir()
            path=root/'absolute_action_t/000000.npy'  # No arm file exists.
            np.save(root/'processed_encoder_t/000000.npy',np.zeros((2,13)))
            np.savez(root/'syncs/000000.npz',encoder_monotonic_ns=[100000000,300000000])
            times,hand,saved=replay.load_hand_actions(path)
            np.testing.assert_allclose(times,[0,.2])
            self.assertFalse(saved)

    def test_device_selection_is_mutually_exclusive(self):
        for options in [['--arm-only','--hand-only'],['--hand-only','--relative'],['--hand-only','--check-start']]:
            with self.subTest(options=options), patch('sys.stderr'), self.assertRaises(SystemExit):
                replay.parse_args(['actions.npy',*options])

    def test_hand_only_keeps_configured_target_limit_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            np.save(root/'hand.npy',np.full((1,13),4.))
            hand_class=Mock()
            with patch('scripts.piper_api.piper_api.PiperAPI') as api, \
                 patch('inference.executor.executor.load_hand_class',return_value=hand_class), \
                 patch('data_collection.collection.hand_common.load_hand_config',return_value=HandConfigFixture()), \
                 contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError,'configured limits'):
                replay.main([str(root/'hand.npy'),'--hand-only','--execute','--hz','5',
                             '--output',str(root/'plan.npz')])
            api.assert_not_called()
            hand_class.assert_not_called()

    def test_relative_main_reanchors_cartesian_input_before_ik(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, poses, hand = self.actions()
            poses[:,:3] += [1., 2., 3.]
            np.save(root/'actions.npy', np.c_[poses,hand])
            with patch.object(replay, 'piper_model', return_value=(self.fk,self.bounds)), \
                 patch('scripts.piper_api.piper_api.PiperAPI') as api, \
                 contextlib.redirect_stdout(io.StringIO()):
                replay.main([str(root/'actions.npy'),'--hz','5','--pose-frame','relative_to_episode_start',
                             '--relative','--start-joints-deg',*map(str,np.degrees(self.seed)),
                             '--output',str(root/'plan.npz')])
            api.assert_not_called()
            with np.load(root/'plan.npz',allow_pickle=False) as saved:
                np.testing.assert_allclose(saved['joints'][0],self.seed,atol=1e-7)
                self.assertEqual(str(saved['replay_mode']),'relative')

    def test_relative_saved_plan_still_rejects_singular_joint_segments(self):
        # Source is valid, but moving its offset toward J5=0 invalidates it.
        source = np.tile(self.seed,(2,1))
        source[1,4] -= .02
        seed = self.seed.copy()
        seed[4] = np.radians(10.5)
        with self.assertRaisesRegex(ValueError,'Wrist singularity'):
            replay.validate_joint_plan(np.array([0.,1.]),source,np.zeros((2,13)),
                                       seed,self.bounds,self.fk,relative=True)

    def test_skip_validation_avoids_preflight_jacobians_for_both_inputs(self):
        original=self.plan()
        with patch.object(replay,'jacobian',side_effect=AssertionError('Jacobian scan should be skipped')):
            plan=self.plan(skip_validation=True)
            saved=replay.validate_joint_plan(original['times'],original['joints'],original['hand'],
                self.seed,self.bounds,self.fk,skip_validation=True)
        self.assertTrue(np.isnan(plan['jacobian_sigma']).all())
        self.assertTrue(np.isnan(saved['jacobian_sigma']).all())
        np.testing.assert_allclose(plan['joints'],original['joints'])

    def test_skip_validation_keeps_joint_and_wrist_limits(self):
        for index,value,message in [(0,3.,'Joint limit'),(4,0.,'Wrist singularity')]:
            seed=self.seed.copy()
            seed[index]=value
            with self.subTest(index=index), self.assertRaisesRegex(ValueError,message):
                replay.plan_actions(*self.actions(),seed,self.bounds,self.fk,skip_validation=True)

    def test_skip_validation_keeps_jump_checks(self):
        times,poses,hand=self.actions()
        poses[1,0]+=.2
        with self.assertRaisesRegex(ValueError,'position jump'):
            self.plan(times,poses,hand,skip_validation=True)

    def test_skip_validation_report_marks_missing_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,poses,hand=self.actions()
            np.save(root/'actions.npy',np.c_[poses,hand])
            with patch.object(replay,'piper_model',return_value=(self.fk,self.bounds)), \
                 patch('scripts.piper_api.piper_api.PiperAPI') as api, \
                 patch.object(replay,'jacobian',side_effect=AssertionError('No preflight Jacobian')), \
                 contextlib.redirect_stdout(io.StringIO()) as log:
                replay.main([str(root/'actions.npy'),'--hz','5','--pose-frame','piper_base',
                             '--skip-validation','--start-joints-deg',*map(str,np.degrees(self.seed)),
                             '--output',str(root/'plan.npz')])
            api.assert_not_called()
            report=json.loads((root/'plan.json').read_text())
            self.assertFalse(report['preflight_jacobian_checked'])
            self.assertIsNone(report['min_jacobian_sigma'])
            self.assertIn('WARNING',log.getvalue())

    def test_freshness_override_accepts_old_feedback_but_not_missing_or_invalid(self):
        api=Mock()
        api.get_joint_state.return_value=SimpleNamespace(timestamp=time.time()-5,joints=self.seed)
        with self.assertRaises(RuntimeError):
            replay.read_fresh_joints(api)
        np.testing.assert_array_equal(replay.read_fresh_joints(api,check_freshness=False),self.seed)
        for state in [None,SimpleNamespace(timestamp=float('nan'),joints=self.seed),
                      SimpleNamespace(timestamp=time.time()-5,joints=[float('nan')]*6)]:
            api.get_joint_state.return_value=state
            with self.assertRaises(RuntimeError):
                replay.read_fresh_joints(api,check_freshness=False)

    def test_freshness_override_keeps_controller_fault_rejection(self):
        api=Mock()
        api.robot.get_arm_status.return_value=SimpleNamespace(timestamp=time.time()-5,
            msg=SimpleNamespace(err_code=0,arm_status='NORMAL'))
        with self.assertRaises(RuntimeError):
            replay.check_status(api)
        replay.check_status(api,check_freshness=False)
        api.robot.get_arm_status.return_value.msg.err_code=1
        with self.assertRaisesRegex(RuntimeError,'controller fault'):
            replay.check_status(api,check_freshness=False)

    def test_playback_propagates_freshness_override(self):
        plan=self.plan()
        api=Mock()
        with patch.object(replay,'read_fresh_joints',return_value=self.seed) as read, \
             patch.object(replay,'check_status') as status,patch.object(replay.time,'sleep'):
            replay.play_plan(plan,api,None,self.bounds,self.fk,replay.Limits(),check_freshness=False)
        self.assertTrue(all(c.kwargs=={'check_freshness':False} for c in read.call_args_list))
        self.assertTrue(all(c.kwargs=={'check_freshness':False} for c in status.call_args_list))


if __name__ == '__main__':
    unittest.main()
