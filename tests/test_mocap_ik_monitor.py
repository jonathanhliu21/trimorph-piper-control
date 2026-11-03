"""Offline live-IK checks with synthetic kinematics; no ROS or arm required."""
from collections import deque
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import cv2

import numpy as np
from scipy.spatial.transform import Rotation

from data_collection.collect_hand_only import main, parse_args
from data_collection.collection.mocap_ik_monitor import (
    PLOT_COLS, PLOT_FILL, PLOT_HEIGHT, PLOTS_TOP, RED_FILL, WIDTH, IKResult, MocapIKChecker, MocapIKMonitor)
from data_collection.collection.shared_state import SharedState, TrackerSample


def calibration():
    return {'vicon_reference': {'run': {
        'vicon_to_piper_rotation': np.eye(3).tolist(),
        'vicon_to_piper_translation_m': [.1, 0., 0.],
        'flange_in_vicon_body_rotation': np.eye(3).tolist(),
        'flange_in_vicon_body_translation_m': [.02, 0., 0.],
    }}}


class MocapIKTests(unittest.TestCase):
    def checker(self):
        return MocapIKChecker(calibration(), [0, 0, 0, 0, 30, 0],
                              model=(lambda q: np.array(q), np.tile([-3., 3.], (6, 1))))

    def sample(self, position=(.2, .1, .3), pitch=.5, lost=False):
        return TrackerSample(1_000_000_000, position,
                             tuple(Rotation.from_euler('xyz', [0, pitch, 0]).as_quat()), lost, 1)

    def test_valid_solution_uses_absolute_world_and_body_offset(self):
        result = self.checker().check(self.sample())
        self.assertTrue(result.ok, result.detail)
        q = np.radians(result.joints_deg)
        expected = np.array([.2, .1, .3]) + [.1, 0, 0] + Rotation.from_euler('y', .5).apply([.02, 0, 0])
        np.testing.assert_allclose(q[:3], expected, atol=1e-6)
        self.assertAlmostEqual(q[4], .5, places=6)

    def test_unreachable_pose_warns_and_keeps_last_valid_seed(self):
        checker = self.checker()
        seed = checker.seed.copy()
        result = checker.check(self.sample(position=(4., .1, .3)))
        self.assertFalse(result.ok)
        self.assertIn('No accurate local IK', result.detail)
        np.testing.assert_array_equal(checker.seed, seed)

    def test_joint_limit_failure_reports_unbounded_joints_for_plotting(self):
        checker = self.checker()
        result = checker.check(self.sample(position=(4., .1, .3)))
        self.assertFalse(result.ok)
        self.assertIn('needs J1=', result.detail)
        self.assertGreater(result.joints_deg[0], np.degrees(3.))
        np.testing.assert_array_equal(checker.out_of_range(result.joints_deg), [1, 0, 0, 0, 0, 0])

    def test_pose_outside_workspace_has_no_joints(self):
        checker = MocapIKChecker(calibration(), [0, 0, 0, 0, 30, 0], model=(
            lambda q: np.r_[np.clip(q[:3], -1, 1), q[3:]], np.tile([-3., 3.], (6, 1))))
        result = checker.check(self.sample(position=(4., .1, .3)))
        self.assertFalse(result.ok)
        self.assertNotIn('needs', result.detail)
        self.assertIsNone(result.joints_deg)

    def test_out_of_range_includes_wrist_band(self):
        checker = self.checker()
        np.testing.assert_array_equal(checker.out_of_range([0, 0, 0, 0, 5, 0]), [0, 0, 0, 0, 1, 0])
        np.testing.assert_array_equal(checker.out_of_range([0, 0, 0, 0, 30, 0]), [0] * 6)

    def monitor(self, recorder=None):
        monitor = MocapIKMonitor.__new__(MocapIKMonitor)
        monitor.checker, monitor.shared, monitor.topic = self.checker(), SharedState(), '/vicon/Yori/Yori'
        monitor.recorder, monitor.cv2 = recorder, cv2
        monitor._lock, monitor._history, monitor._result = threading.Lock(), deque(), None
        monitor._episode, monitor._stopped, monitor.last_episode = None, [], None
        monitor.episodes_in_range = monitor.episodes_scored = 0
        return monitor

    @staticmethod
    def cell_fill(image, joint):
        row, col = divmod(joint, PLOT_COLS)
        return tuple(image[PLOTS_TOP + row * PLOT_HEIGHT + PLOT_HEIGHT - 8, col * (WIDTH // PLOT_COLS) + 8])

    def test_plot_fill_follows_latest_solution_of_each_joint(self):
        monitor = self.monitor()
        s = 1_000_000_000
        good, bad = (0, 30, 0, 0, 30, 0), (200, 30, 0, 0, 30, 0)
        for ns, joints in ((5 * s, bad), (20 * s, good), (21 * s, good)):
            monitor._record(IKResult(ns, True, 'OK', joints))
        image = monitor.render(22 * s)  # Not recording: the old J1 excursion is outside the last 10 s.
        self.assertEqual(self.cell_fill(image, 0), PLOT_FILL)
        monitor._record(IKResult(22 * s, False, 'limit', bad))
        image = monitor.render(22 * s)
        self.assertEqual(self.cell_fill(image, 0), RED_FILL)
        for joint in range(1, 6):
            self.assertEqual(self.cell_fill(image, joint), PLOT_FILL)
        monitor._record(IKResult(23 * s, False, 'OBJECT LOST'))  # No joints: keeps the last solution's state.
        self.assertEqual(self.cell_fill(monitor.render(23 * s), 0), RED_FILL)
        monitor._record(IKResult(24 * s, True, 'OK', good))  # Back in range clears the fill.
        self.assertEqual(self.cell_fill(monitor.render(24 * s), 0), PLOT_FILL)

    def test_recording_plots_whole_episode_since_start(self):
        session = SimpleNamespace(start_ns=10 * 1_000_000_000)
        monitor = self.monitor(SimpleNamespace(current=lambda: session))
        monitor._record(IKResult(5 * 1_000_000_000, False, 'limit', (200, 30, 0, 0, 30, 0)))
        self.assertEqual(self.cell_fill(monitor.render(40 * 1_000_000_000), 0), PLOT_FILL)
        monitor._record(IKResult(12 * 1_000_000_000, False, 'limit', (200, 30, 0, 0, 30, 0)))
        self.assertEqual(self.cell_fill(monitor.render(13 * 1_000_000_000), 0), RED_FILL)
        monitor._record(IKResult(38 * 1_000_000_000, True, 'OK', (0, 30, 0, 0, 30, 0)))
        t0, t1, text = monitor.plot_span(40 * 1_000_000_000)
        self.assertEqual((t0, t1), (10 * 1_000_000_000, 40 * 1_000_000_000))
        self.assertIn('REC 30.0 s', text)
        self.assertEqual(self.cell_fill(monitor.render(40 * 1_000_000_000), 0), PLOT_FILL)  # Back in range; the J1 excursion remains as red trace.

    def test_score_counts_saved_episodes_with_all_joints_in_range(self):
        s = 1_000_000_000
        recorder = SimpleNamespace(session=None)
        recorder.current = lambda: recorder.session
        monitor = self.monitor(recorder)
        good, bad = (0, 30, 0, 0, 30, 0), (0, 30, 0, 0, 5, 0)

        def episode(index, start, stop, results, reason='foot_pedal'):
            session = SimpleNamespace(start_ns=start * s, stop_ns=None, stop_reason='',
                                      paths=SimpleNamespace(episode_index=index))
            recorder.session = session
            monitor.track_episodes(start * s)
            for t, result in results:
                monitor._record(IKResult(int(t * s), *result))
            recorder.session = None
            session.stop_ns, session.stop_reason = stop * s, reason
            monitor.track_episodes(stop * s)  # Too early: IK on the last samples may still be running.
            self.assertIn(session, monitor._stopped)
            monitor.track_episodes(int((stop + .5) * s))

        episode(0, 0, 2, [(.5, (True, 'OK', good)), (1.5, (True, 'OK', good)),
                          (2.5, (False, 'after stop', bad))])
        self.assertEqual((monitor.episodes_in_range, monitor.episodes_scored), (1, 1))
        self.assertEqual(monitor.last_episode, (0, True, 'all joints in range'))
        episode(1, 10, 12, [(10.5, (True, 'OK', good)), (11, (False, 'wrist', bad))])
        self.assertEqual((monitor.episodes_in_range, monitor.episodes_scored), (1, 2))
        self.assertEqual(monitor.last_episode, (1, False, 'out of range: J5'))
        episode(2, 20, 22, [(21, (False, 'wrist', bad))], reason='foot_pedal_delete')
        self.assertEqual(monitor.episodes_scored, 2)  # Deleted episodes are not counted.
        episode(3, 30, 32, [(30.5, (True, 'OK', good)), (31, (False, 'OBJECT LOST'))])
        self.assertEqual(monitor.episodes_in_range, 2)  # Tracking loss alone is not out of range.
        episode(4, 40, 42, [(41, (False, 'No accurate local IK within joint margins: 9 mm'))])
        self.assertEqual((monitor.episodes_in_range, monitor.episodes_scored), (2, 4))
        self.assertEqual(monitor.last_episode[2], 'out of range: 1 unreachable')
        monitor.render(43 * s)

    def test_wrist_and_lost_tracking_are_not_green(self):
        result = self.checker().check(self.sample(pitch=0.))
        self.assertFalse(result.ok)
        self.assertIn('Wrist singularity', result.detail)
        self.assertEqual(self.checker().check(self.sample(lost=True)).detail, 'OBJECT LOST')

    def test_calibration_without_vicon_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'vicon_reference'):
            MocapIKChecker({}, [0] * 6)

    def test_status_cannot_show_old_green_when_tracking_is_lost_or_result_stale(self):
        monitor = MocapIKMonitor.__new__(MocapIKMonitor)
        monitor.shared = SharedState()
        monitor.topic = '/vicon/Yori/Yori'
        monitor._lock = threading.Lock()
        monitor._result = IKResult(1_000_000_000, True, 'OK')
        self.assertEqual(monitor.status(1_000_000_000)[0], 'NO IK RESULT')
        monitor.shared.set_tracker_pose(monitor.topic, 1_000_000_000, (.1, .2, .3), (0, 0, 0, 1))
        self.assertEqual(monitor.status(1_000_000_000)[0], 'IK OK (local branch)')
        monitor.shared.set_tracker_pose(monitor.topic, 2_000_000_000, (.1, .2, .3), (0, 0, 0, 1))
        self.assertEqual(monitor.status(2_000_000_000)[0], 'IK RESULT STALE')
        monitor.shared.set_tracker_pose(monitor.topic, 2_000_000_000, (0, 0, 0), (0, 0, 0, 1), lost=True)
        self.assertEqual(monitor.status(2_000_000_000)[1], 'OBJECT LOST')

    def test_cli_requires_mocap_and_propagates_settings(self):
        for options in [
            ['--no-mocap'],
            ['--mocap-topic', '/other'],
            ['--mocap-ik-seed-deg', 'nan', '0', '0', '0', '0', '0'],
        ]:
            with self.subTest(options=options), patch('sys.argv', [
                    'collect', '--mocap-ik-calibration', 'cal.json', *options]), \
                    patch('sys.stderr'), self.assertRaises(SystemExit):
                parse_args()
        with patch('sys.argv', ['collect', '--use-mocap', '--no-monitor',
                               '--mocap-ik-calibration', 'cal.json']), \
                patch('data_collection.collect_hand_only.run_hand_only', return_value=0) as run:
            main()
        config = run.call_args.args[0]
        self.assertEqual(str(config.mocap_ik_calibration), 'cal.json')
        self.assertEqual(config.mocap_ik_topic, '/vicon/Yori/Yori')
        self.assertTrue(config.use_mocap)


if __name__ == '__main__':
    unittest.main()
