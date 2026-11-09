"""Hardware-free coverage of paired raw episodes, conversion and replay comparisons."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from data_collection import replay_actions
from data_collection.collection.episode import episode_paths
from scripts.quest_piper.collect import Episode, Recorder, FootPedalController, parse_args as collect_args, piper_worker
from scripts.quest_piper.common import interpolate, read_poses, write_csv, write_json, pose_rows
from scripts.quest_piper.postprocess import process
from scripts.quest_piper.replay import ReplayComparison
from scripts.quest_piper import replay as pipeline_replay
from scripts.quest.visualize_quest_accuracy_piper import PiperHistory


class QuestPiperPipelineTests(unittest.TestCase):
    def test_collection_defaults_to_pedal_except_timed_or_explicit_keyboard_mode(self):
        self.assertTrue(collect_args(['--dataset-root', 'unused']).pedal)
        self.assertFalse(collect_args(['--dataset-root', 'unused', '--episode-s', '1']).pedal)
        self.assertFalse(collect_args(['--dataset-root', 'unused', '--no-pedal']).pedal)

    def test_pedal_matches_ready_delay_cancel_save_on_release_and_hold_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            args = collect_args(['--dataset-root', directory])
            recorder, pedal = Recorder(Path(directory)), Mock()
            pedal.poll.return_value = [1]
            controller = FootPedalController(args, recorder, pedal)
            with patch('scripts.quest_piper.collect.time.monotonic', return_value=10.):
                controller.poll()
            self.assertIsNone(recorder.current())
            self.assertAlmostEqual(controller.pending_start_deadline, 10.2)
            controller.poll()  # another press cancels the pending start
            self.assertIsNone(controller.pending_start_deadline)
            with patch('scripts.quest_piper.collect.time.monotonic', return_value=11.):
                controller.poll()
            pedal.poll.return_value = []
            with patch('scripts.quest_piper.collect.time.monotonic', return_value=11.3):
                controller.poll()
            first = recorder.current()
            self.assertIsNotNone(first)
            # Releasing the initial start press must not save the episode.
            pedal.current_hold_s.return_value = None
            controller.poll()
            self.assertIs(recorder.current(), first)
            pedal.poll.return_value = [1]
            pedal.current_hold_s.return_value = .1
            controller.poll()
            self.assertIs(recorder.current(), first)
            pedal.poll.return_value = []
            pedal.current_hold_s.return_value = None
            controller.poll()
            self.assertIsNone(recorder.current())
            self.assertTrue(first.paths.metadata_path.exists())
            self.assertEqual(json.loads(first.paths.metadata_path.read_text())['stop_reason'], 'foot_pedal')
            second = recorder.start()
            pedal.poll.return_value = [1]
            pedal.current_hold_s.return_value = 2.1
            controller.poll()
            self.assertIsNone(recorder.current())
            self.assertTrue(second.closed)
            self.assertFalse(second.paths.episode_dir.exists())

    def test_replay_entrypoint_forces_arm_only_and_planning_has_no_execution_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, calibration, _, _ = self.fixture(root)
            process(root, 0, calibration)
            output = root/'test_plan'
            with patch.object(replay_actions, 'main', return_value=0) as main:
                pipeline_replay.main([str(root), '--no-view', '--output-dir', str(output)])
            args = main.call_args.args[0]
            self.assertIn('--arm-only', args)
            self.assertIn('--warn-margins', args)
            self.assertIn('--warn-motion-limits', args)
            self.assertNotIn('--execute', args)
            self.assertIsNone(main.call_args.kwargs['playback_observer'])
            self.assertEqual(json.loads((output/'comparison.json').read_text())['samples'], 0)
            with patch.object(replay_actions, 'main', return_value=0) as main:
                pipeline_replay.main([str(root), '--no-view', '--strict-limits', '--output-dir', str(root/'strict_plan')])
            self.assertNotIn('--warn-margins', main.call_args.args[0])
            self.assertNotIn('--warn-motion-limits', main.call_args.args[0])

    def test_shared_visualizer_draws_original_trajectory_headlessly(self):
        import cv2
        from scripts.quest_piper.visualizer import PipelineView
        from scripts.quest.visualize_quest_accuracy_piper import Comparison, parse_args
        shared = PiperHistory()
        c = Comparison(parse_args([]))
        c.anchor = (np.zeros(3), np.array([0,0,0,1]), np.zeros(3), np.eye(3))
        with patch.object(cv2, 'namedWindow'), patch.object(cv2, 'setMouseCallback'):
            view = PipelineView(shared, c.args, c)
        view.original = np.array([[0.,0,0],[.1,.2,.3]])
        view.banner = 'RECORDING episode_000000'
        frame = view.draw({}, 1_000_000_000, None)
        self.assertEqual(frame.shape, (900,1280,3))
        self.assertTrue(np.any(np.all(frame == (220,80,220), axis=2)))

    def fixture(self, root, bias=0.):
        root = Path(root)
        raw = root/'raw'/'episode_000000'
        stamps = 1_000_000_000 + np.arange(31, dtype=np.int64)*10_000_000
        rotations = Rotation.from_euler('xyz', np.c_[np.linspace(0, .2, 31), np.linspace(0, -.3, 31), np.zeros(31)])
        q = np.c_[np.linspace(.1, .13, 31), np.zeros(31), np.full(31, .2)]
        lever, origin = np.array([.03, 0, .02]), np.array([.2, .1, -.1])
        p = q + rotations.apply(lever) + origin
        lag = .02
        write_csv(raw/'piper.csv', pose_rows(stamps, p, rotations))
        write_csv(raw/'quest.csv', [dict(sample_idx=i, monotonic_ns=int(t)+20_000_000, seq=i,
                    **dict(zip(('px','py','pz','qx','qy','qz','qw'), map(float, np.r_[pos+[bias,0,0], quat]))))
                    for i, (t, pos, quat) in enumerate(zip(stamps, q, rotations.as_quat()))])
        write_json(raw/'metadata.json', dict(status='complete', started_monotonic_ns=int(stamps[0])))
        cal = dict(direction='quest_to_piper', tracker='quest', linear_matrix=np.eye(3).tolist(),
                   orientation=dict(rotation_matrix=np.eye(3).tolist(), controller_rotation=dict(matrix=np.eye(3).tolist())),
                   tracked_point_offset=dict(frame='piper_flange', vector=lever.tolist()),
                   position_offset=origin.tolist(), latency_s=lag)
        calibration = root/'calibration.json'
        write_json(calibration, cal)
        return raw, calibration, stamps, p

    def test_postprocess_uses_latency_offset_and_original_without_fake_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, calibration, stamps, p = self.fixture(root)
            before = (raw/'quest.csv').read_bytes()
            report = process(root, 0, calibration)
            self.assertLess(report['comparison']['max']['position_mm'], 1e-8)
            self.assertLess(report['comparison']['max']['orientation_deg'], 1e-8)
            mapped = np.load(root/'absolute_action_t'/'000000.npy')
            original = np.load(root/'original_action_t'/'000000.npy')
            self.assertEqual(mapped.shape, (31, 6))
            np.testing.assert_allclose(mapped, original, atol=1e-7)
            np.testing.assert_allclose(mapped[:, :3], p, atol=1e-7)
            with np.load(root/'syncs'/'000000.npz') as sync:
                np.testing.assert_array_equal(sync['pose_monotonic_ns'], stamps)
                self.assertNotIn('encoder_monotonic_ns', sync)
                self.assertEqual(str(sync['pose_frame']), 'piper_base')
            self.assertFalse((root/'processed_encoder_t').exists())
            self.assertEqual(before, (raw/'quest.csv').read_bytes())
            with self.assertRaises(FileExistsError):
                process(root, 0, calibration)

    def test_absolute_mode_exposes_bias_that_relative_anchor_cancels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, calibration, _, _ = self.fixture(root, bias=.01)
            relative = process(root, 0, calibration)
            absolute = process(root, 0, calibration, absolute=True, overwrite=True)
            self.assertLess(relative['comparison']['rmse']['position_mm'], 1e-8)
            self.assertAlmostEqual(absolute['comparison']['rmse']['position_mm'], 10., places=6)

    def test_interior_reference_gaps_refused_before_outputs_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, calibration, _, _ = self.fixture(root)
            lines = (raw/'piper.csv').read_text().splitlines()
            (raw/'piper.csv').write_text('\n'.join(lines[:11]+lines[20:])+'\n')
            with self.assertRaisesRegex(ValueError, 'interior tracking gap'):
                process(root, 0, calibration)
            self.assertFalse((root/'absolute_action_t').exists())

    def test_interpolation_exact_pose_allowed_next_to_gap_but_no_extrapolation(self):
        stamps = np.array([100, 200, 1_000_000_000], np.int64)
        p, r, ok = interpolate(stamps, np.zeros((3,3)), Rotation.identity(3), [99, 100, 150, 300, 1_000_000_001])
        np.testing.assert_array_equal(ok, [False, True, True, False, False])

    def test_recorder_reuses_episode_layout_and_marks_hand_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory))
            session = recorder.start()
            for i in range(2):
                stamp = session.start_ns + i*10_000_000
                session.append_quest(dict(monotonic_ns=stamp, px=0., py=0., pz=0., qx=0., qy=0., qz=0., qw=1.))
                session.append_piper(dict(monotonic_ns=stamp, x=.2, y=0., z=.3, roll=0., pitch=0., yaw=0.))
            recorder.stop('operator')
            meta = json.loads(session.paths.metadata_path.read_text())
            self.assertEqual(meta['status'], 'complete')
            self.assertIsNone(meta['paths']['leader_hand_csv'])
            self.assertFalse(meta['hand']['enabled'])
            self.assertFalse(session.paths.leader_hand_path.exists())
            self.assertEqual(recorder.start().paths.episode_index, 1)

    def test_passive_worker_uses_fk_and_skips_cached_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder, shared = Recorder(Path(directory)), PiperHistory()
            session = recorder.start()
            api = Mock()
            api.connect.return_value = True
            api.robot.fk.return_value = [.2, .1, .3, 0., .4, 0.]
            api.get_state.return_value = {}
            api.get_eef_state.return_value = None
            calls = 0
            def joints():
                nonlocal calls
                calls += 1
                if calls >= 3:
                    shared.shutdown.set()
                return SimpleNamespace(timestamp=1., hz=200., joints=[0.]*6)
            api.get_joint_state.side_effect = joints
            args = SimpleNamespace(firmware='v183', interface='socketcan', channel='can0', rate_hz=200.)
            with patch('scripts.piper_api.piper_api.PiperAPI', return_value=api) as factory:
                piper_worker(args, shared, recorder, init_done=threading.Event())
            self.assertEqual(len(session.piper), 1)
            self.assertEqual(session.piper[0]['x'], .2)
            self.assertEqual(session.piper[0]['fk_pitch'], .4)
            self.assertIsNone(factory.call_args.kwargs['payload'])
            api.enable.assert_not_called(); api.move_joint.assert_not_called(); api.move_eef.assert_not_called()
            api.disconnect.assert_called_once()

    def test_replay_comparison_uses_actual_time_and_excludes_approach(self):
        with tempfile.TemporaryDirectory() as directory:
            original = np.array([[0., 0, 0, 0, 0, 0], [1., 0, 0, 0, 0, 0]])
            comparison = ReplayComparison(original, np.array([1_000_000_000, 2_000_000_000]), .5, Path(directory), show=False)
            comparison.fk = lambda q: q
            plan = dict(times=np.array([0., 1., 3.]), poses=np.zeros((3,6)), joints=np.zeros((3,6)))
            # One second approach; actual elapsed=2 -> source elapsed=.5, original X=.5.
            comparison(plan, 0, 0., 100, np.zeros(6))
            self.assertEqual(comparison.comparison.count, 0)
            comparison(plan, 1, 2., 200, np.array([.5,0,0,0,0,0]))
            c = comparison.comparison
            self.assertEqual(c.count, 1)
            np.testing.assert_allclose(c.latest[1], [.5,0,0])
            np.testing.assert_allclose(c.latest[3], [.5,0,0])
            np.testing.assert_allclose(c.latest[-1], [0,0])
            np.testing.assert_allclose(c.squares, [0,0])
            comparison.finish(True)
            report = json.loads((Path(directory)/'comparison.json').read_text())
            self.assertEqual(report['trajectory_vs_original']['samples'], 1)
            self.assertEqual(report['trajectory_vs_original']['max']['position_mm'], 0.)
            self.assertEqual(report['trajectory_vs_scheduled_target']['max']['position_mm'], 500.)

    def test_existing_replay_observer_receives_measured_feedback(self):
        api, observer = Mock(), Mock()
        measured = np.array([0., .5, -.5, 0., .5, 0.])
        plan = dict(times=np.array([0.]), joints=measured[None,:], hand=np.empty((1,0)))
        with patch.object(replay_actions, 'read_fresh_joints', return_value=measured), \
             patch.object(replay_actions, 'check_joints'), patch.object(replay_actions, 'check_status'):
            replay_actions.play_plan(plan, api, None, None, None, replay_actions.Limits(), observer=observer)
        observer.assert_called_once()
        np.testing.assert_array_equal(observer.call_args.args[-1], measured)
        api.move_joint.assert_called_once_with(measured.tolist(), mode='smooth')


if __name__ == '__main__':
    unittest.main()
