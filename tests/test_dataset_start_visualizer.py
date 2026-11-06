import tempfile
import unittest
from pathlib import Path
import subprocess
import sys
import textwrap
import struct
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.dataset.visualize_start_positions import PiperFeedback, StartViewer, home_pose, load_starts


class DatasetStartsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'absolute_action_t').mkdir()
        (self.root / 'syncs').mkdir()

    def episode(self, name, *, frame='piper_base', index=1, poses=None):
        if poses is None:
            poses = np.array([[.1, .2, .3, 0, 0, 0], [.4, .5, .6, 0, 0, np.pi]])
        np.save(self.root / 'absolute_action_t' / f'{name}.npy', poses)
        np.savez(self.root / 'syncs' / f'{name}.npz', pose_frame=frame, pose_idx=[index])

    def test_one_start_per_episode_in_numeric_order(self):
        self.episode('10', poses=np.ones((100, 6)))
        self.episode('2')
        starts = load_starts(self.root)
        np.testing.assert_array_equal(starts.episodes, [2, 10])
        np.testing.assert_allclose(starts.poses[:, 0], [.1, 1])

    def test_sync_uses_camera_mapping_instead_of_first_pose(self):
        self.episode('000001')
        np.testing.assert_allclose(load_starts(self.root).poses[0, :3], [.1, .2, .3])
        np.testing.assert_allclose(load_starts(self.root, 'sync').poses[0], [.4, .5, .6, 0, 0, np.pi])

    def test_relative_frame_cannot_be_compared_to_absolute_home(self):
        self.episode('000001', frame='relative_to_episode_start')
        with self.assertRaisesRegex(ValueError, 'pose_frame=piper_base'):
            load_starts(self.root)

    def test_bad_sync_index_and_nonfinite_start_are_rejected(self):
        self.episode('000001', index=-1)
        with self.assertRaisesRegex(ValueError, 'valid start index'):
            load_starts(self.root, 'sync')
        self.episode('000001', poses=np.full((1, 6), np.nan))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            load_starts(self.root)

    def test_cli_offline_export_and_interactive_view_do_not_access_hardware(self):
        self.episode('000001')
        # A fresh interpreter audits imports too. Fail on even an attempted
        # device/socket access, process launch, or robot factory invocation.
        code = textwrap.dedent('''
            import os
            import runpy
            import sys
            from unittest.mock import patch

            attempts = []
            def audit(event, args):
                device = (event == 'open' and isinstance(args[0], (str, bytes, os.PathLike))
                          and os.fsdecode(args[0]).startswith('/dev/')
                          and os.fsdecode(args[0]) not in ('/dev/null', '/dev/urandom', '/dev/random'))
                if device or event in ('socket.__new__', 'socket.connect', 'socket.bind',
                                       'subprocess.Popen', 'os.system'):
                    attempts.append(event)
                    raise AssertionError('Unexpected hardware/process access: ' + event)
            sys.addaudithook(audit)

            import cv2
            from pyAgxArm import AgxArmFactory
            def no_robot(*args, **kwargs):
                attempts.append('create_arm')
                raise AssertionError('Unexpected robot creation')

            script, dataset, output = sys.argv[1:]
            with patch.object(AgxArmFactory, 'create_arm', side_effect=no_robot):
                sys.argv = [script, dataset, '--home', 'home_unscrew1', '--output', output]
                runpy.run_path(script, run_name='__main__')
                # Run the actual window loop with GUI calls replaced so this
                # regression check also works on machines without a display.
                with patch.multiple(cv2, namedWindow=lambda *a: None,
                                    resizeWindow=lambda *a: None,
                                    setMouseCallback=lambda *a: None,
                                    imshow=lambda *a: None, waitKey=lambda *a: ord('q'),
                                    destroyAllWindows=lambda *a: None):
                    sys.argv = [script, dataset, '--home', 'home_unscrew1', '--sample', 'sync']
                    runpy.run_path(script, run_name='__main__')
            assert not attempts, attempts
        ''')
        script = Path(__file__).resolve().parents[1] / 'scripts/dataset/visualize_start_positions.py'
        output = self.root / 'preview.png'
        result = subprocess.run([sys.executable, '-c', code, str(script), str(self.root), str(output)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(output.is_file())


class ReceiveOnlyBus:
    """Any call beyond receiving and closing fails, including all send methods."""
    def __init__(self):
        self.messages = []
        self.closed = False

    def recv(self, timeout):
        return self.messages.pop(0) if self.messages else None

    def shutdown(self):
        self.closed = True

    def message(self, pair, angles, stamp=100.):
        self.messages.append(SimpleNamespace(arbitration_id=0x2A5 + pair,
                             data=struct.pack('>ii', *angles), timestamp=stamp,
                             is_extended_id=False, is_error_frame=False, is_remote_frame=False))


class PiperFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.bus = ReceiveOnlyBus()
        self.feedback = PiperFeedback('unused', bus=self.bus)

    def fill(self, stamp=100.):
        for pair, values in enumerate(((-49000, 99000), (-56000, 17000), (28000, 26000))):
            self.bus.message(pair, values, stamp)

    def test_all_six_joints_required_and_converted_from_signed_millidegrees(self):
        self.bus.message(0, (-49000, 99000))
        self.assertIsNone(self.feedback.poll(now=100.)[0])
        self.fill()
        pose, status = self.feedback.poll(now=100.1)
        np.testing.assert_allclose(pose, home_pose([-49, 99, -56, 17, 28, 26]))
        self.assertIn('read-only', status)
        self.feedback.close()
        self.assertTrue(self.bus.closed)

    def test_one_stale_pair_hides_pose_even_with_fresh_other_joints(self):
        self.fill()
        self.assertIsNotNone(self.feedback.poll(now=100.)[0])
        self.bus.message(1, (0, 0), stamp=101.)
        self.bus.message(2, (0, 0), stamp=101.)
        self.assertIsNone(self.feedback.poll(now=101.)[0])

    def test_old_queued_and_future_feedback_are_rejected(self):
        for timestamp in (90., 110.):
            self.fill(timestamp)
            self.assertIsNone(self.feedback.poll(now=100.)[0])

    def test_command_frames_and_bad_frames_cannot_supply_feedback(self):
        self.fill()
        self.bus.messages[0].arbitration_id = 0x155
        self.bus.messages[1].is_remote_frame = True
        self.bus.messages[2].data = b'bad'
        self.assertIsNone(self.feedback.poll(now=100.)[0])

    def test_live_view_updates_then_hides_stale_marker(self):
        starts = SimpleNamespace(episodes=np.array([0, 1]),
                                 poses=np.array([[.2, -.2, .1, 0, 0, 0], [.3, -.3, .2, 0, 0, 0]]))
        viewer = StartViewer(starts, home_pose([-49, 99, -56, 17, 28, 26]), 'home', 'fixture', 'first')
        viewer.live_enabled = True
        self.fill()
        viewer.update_live(*self.feedback.poll(now=100.))
        self.assertIsNotNone(viewer.live_pose)
        fresh = viewer.render()
        viewer.update_live(*self.feedback.poll(now=101.))
        self.assertIsNone(viewer.live_pose)
        self.assertFalse(np.array_equal(fresh, viewer.render()))

    def test_cli_live_only_receives_and_closes_bus(self):
        # Exercise main's live branch, including cleanup, without real CAN/GUI.
        from scripts.dataset.visualize_start_positions import main
        self.fill(stamp=time.time())
        starts = SimpleNamespace(episodes=np.array([0]), poses=np.array([[.2, -.2, .1, 0, 0, 0]]))
        with patch('can.Bus', return_value=self.bus) as factory, \
             patch('scripts.dataset.visualize_start_positions.load_starts', return_value=starts), \
             patch.object(sys, 'argv', ['viewer', 'dataset', '--live', '--piper-channel', 'can9']), \
             patch.multiple('cv2', namedWindow=lambda *a: None, resizeWindow=lambda *a: None, setMouseCallback=lambda *a: None,
                            imshow=lambda *a: None, waitKey=lambda *a: ord('q'), destroyAllWindows=lambda *a: None):
            main()
        self.assertTrue(self.bus.closed)
        self.assertEqual(factory.call_args.kwargs['channel'], 'can9')


class OrientationDisplayTests(unittest.TestCase):
    def test_roll_wrap_does_not_split_nearby_orientations(self):
        rotations = Rotation.from_euler('xyz', [[179, 0, 20], [-179, 0, 20]], degrees=True)
        starts = SimpleNamespace(episodes=np.array([0, 1]),
                                 poses=np.c_[np.zeros((2, 3)), rotations.as_rotvec()])
        viewer = StartViewer(starts, starts.poses[0], 'test', 'fixture', 'first')
        self.assertAlmostEqual(np.ptp(viewer.rpy_values[:, 0]), 2.)
        live = Rotation.from_euler('xyz', [-178, 0, 20], degrees=True)
        self.assertAlmostEqual(abs(viewer.display_rpy(live)[0] - viewer.rpy_values[1, 0]), 1.)

    def test_display_rpy_uses_euler_degrees_instead_of_rotvec_components(self):
        rotation = Rotation.from_euler('xyz', [159, -1, 94], degrees=True)
        pose = np.r_[np.zeros(3), rotation.as_rotvec()]
        starts = SimpleNamespace(episodes=np.array([0]), poses=pose[None])
        viewer = StartViewer(starts, pose, 'test', 'fixture', 'first')
        np.testing.assert_allclose(viewer.rpy_values[0], [159, -1, 94], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
