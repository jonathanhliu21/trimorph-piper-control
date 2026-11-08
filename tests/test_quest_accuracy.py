"""Hardware-free checks of the accuracy viewer's measurement and rendering."""
import unittest
from unittest.mock import patch

import numpy as np

from scripts.quest.visualize_quest_accuracy import (
    AccuracyView, PoseHistory, fit_alignment, fit_offset_alignment, frozen_status, parse_args,
    signal_status, synchronized_position, synchronization_issue,
)
from data_collection.collection.shared_state import TrackerSample


def sample(t, p, lost=False):
    return TrackerSample(round(t * 1e9), tuple(p), (0., 0., 0., 1.), lost, 1)


class AccuracyTests(unittest.TestCase):
    def offset_data(self, count=180, seed=12):
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(seed)
        rotations = Rotation.from_rotvec(rng.normal(size=(count, 3)) * .6).as_matrix()
        mocap = rng.normal(size=(count, 3)) * .15 + [.4, .2, .8]
        matrix = Rotation.from_euler('xyz', [.2, -.4, .7]).as_matrix() @ np.diag([1., 1., -1.])
        offset = np.array([.3, -.2, .5])
        body_offset = np.array([.08, -.04, .12])
        quest = (mocap + rotations @ body_offset - offset) @ matrix
        return quest, mocap, rotations, matrix, offset, body_offset

    def test_joint_fit_recovers_body_offset_and_predicts_unseen_rotations(self):
        q, m, r, expected_a, expected_t, expected_d = self.offset_data()
        matrix, offset, body_offset, rms = fit_offset_alignment(q, m, r)
        np.testing.assert_allclose(matrix, expected_a, atol=1e-7)
        np.testing.assert_allclose(offset, expected_t, atol=1e-7)
        np.testing.assert_allclose(body_offset, expected_d, atol=1e-7)
        self.assertLess(rms, 1e-7)
        q2, m2, r2, *_ = self.offset_data(seed=77)
        np.testing.assert_allclose(q2 @ matrix.T + offset, m2 + r2 @ body_offset, atol=1e-7)
        # Also recover a transform without a handedness change.
        q[:, 0] *= -1
        matrix, offset, body_offset, rms = fit_offset_alignment(q, m, r)
        self.assertGreater(np.linalg.det(matrix), 0)
        np.testing.assert_allclose(body_offset, expected_d, atol=1e-7)
        self.assertLess(rms, 1e-7)

    def test_joint_fit_rejects_fixed_orientation_and_single_axis_rotation(self):
        from scipy.spatial.transform import Rotation
        q, m, r, *_ = self.offset_data()
        for rotations in (np.tile(np.eye(3), (len(q), 1, 1)),
                          Rotation.from_euler('z', np.linspace(-1, 1, len(q))[:, None]).as_matrix()):
            with self.assertRaisesRegex(ValueError, 'TWO axes'):
                fit_offset_alignment(q, m, rotations)

    def test_joint_fit_tolerates_position_noise_and_isolated_outliers(self):
        q, m, r, matrix, offset, expected = self.offset_data()
        rng = np.random.default_rng(8)
        q += rng.normal(scale=.0003, size=q.shape)
        q[:4] += .04
        _, _, body_offset, _ = fit_offset_alignment(q, m, r)
        np.testing.assert_allclose(body_offset, expected, atol=.002)

    def test_pose_interpolates_rotation_before_applying_body_offset(self):
        from scipy.spatial.transform import Rotation
        a = sample(1, [0, 0, 0])
        b = TrackerSample(1_020_000_000, (0., 0., 0.),
                          tuple(Rotation.from_euler('z', 90, degrees=True).as_quat()), False, 2)
        p = synchronized_position([a, b], 1_010_000_000, .1, [1., 0., 0.])
        np.testing.assert_allclose(p, [2 ** -.5, 2 ** -.5, 0.], atol=1e-12)

    def test_calibration_uses_new_offset_for_subsequent_measurements(self):
        from scipy.spatial.transform import Rotation
        view, shared = self.make_view()
        self.assertEqual(view.args.calibration_seconds, 20.)
        view.calibrate(1_000_000_000)
        q, m, r, _, _, expected = self.offset_data()
        for i in range(len(q)):
            stamp = 1_010_000_000 + i * 20_000_000
            shared.set_tracker_pose('quest', stamp, q[i], [0, 0, 0, 1])
            shared.set_tracker_pose('/body', stamp, m[i], Rotation.from_matrix(r[i]).as_quat())
        view.ingest(shared.snapshot(), 21_000_000_000)
        np.testing.assert_allclose(view.body_offset, expected, atol=1e-7)
        self.assertIsNone(view.calibration_failure)
        stamp = 21_010_000_000
        shared.set_tracker_pose('quest', stamp, q[0], [0, 0, 0, 1])
        shared.set_tracker_pose('/body', stamp, m[0], Rotation.from_matrix(r[0]).as_quat())
        view.ingest(shared.snapshot(), stamp)
        self.assertLess(view.last_error, 1e-7)
        # R clears the display but preserves the fitted physical offset.
        view.reset_history()
        np.testing.assert_allclose(view.body_offset, expected, atol=1e-7)

    def test_manual_body_offset_remains_supported(self):
        from scipy.spatial.transform import Rotation
        view, shared = self.make_view()
        q, m, r, matrix, offset, expected = self.offset_data()
        view.args.mocap_offset = expected.copy()
        view.calibrate(1_000_000_000)
        for i in range(len(q)):
            stamp = 1_010_000_000 + i * 20_000_000
            shared.set_tracker_pose('quest', stamp, q[i], [0, 0, 0, 1])
            shared.set_tracker_pose('/body', stamp, m[i], Rotation.from_matrix(r[i]).as_quat())
        view.ingest(shared.snapshot(), 21_000_000_000)
        self.assertIsNone(view.calibration_failure)
        np.testing.assert_allclose(view.matrix, matrix, atol=1e-7)
        np.testing.assert_allclose(view.offset, offset, atol=1e-7)
        np.testing.assert_allclose(view.body_offset, expected, atol=1e-7)

    def test_display_origin_stays_fixed_when_history_rolls_over(self):
        from collections import deque
        view, shared = self.make_view()
        view.trail = deque(maxlen=2)
        view.matrix, view.offset = np.eye(3), np.zeros(3)
        for i in range(4):
            for name in ('quest', '/body'):
                shared.set_tracker_pose(name, 1_000_000_000 + i * 10_000_000,
                                        [i * .01, 0, 0], [0, 0, 0, 1])
        view.render(1_030_000_000)
        self.assertEqual(len(view.trail), 2)
        np.testing.assert_allclose(view.trail[0][2], [.02, 0, 0])
        np.testing.assert_allclose(view.display_origin, [0, 0, 0])
        view.reset_history()
        self.assertIsNone(view.display_origin)

    def make_view(self):
        import cv2
        shared = PoseHistory()
        with patch.object(cv2, 'namedWindow'), patch.object(cv2, 'setMouseCallback'):
            view = AccuracyView(shared, parse_args(['--mocap-topic', '/body']))
        return view, shared

    def test_healthy_streams_still_require_calibration(self):
        view, shared = self.make_view()
        for name in ('quest', '/body'):
            shared.set_tracker_pose(name, 2_000_000_000, [1, 1, 1], [0, 0, 0, 1])
        view.render(2_000_000_000)
        self.assertEqual(view.paired_count, 1)
        self.assertIn('CALIBRATION REQUIRED', view.unavailable_reason(['OK', 'OK'], True))
        self.assertIsNone(view.last_error)

    def test_failed_calibration_is_reported_separately_from_sync(self):
        view, shared = self.make_view()
        view.calibrate(1_000_000_000)
        for i in range(30):
            for name in ('quest', '/body'):
                shared.set_tracker_pose(name, 1_000_000_000 + i * 10_000_000,
                                        [1, 1, 1], [0, 0, 0, 1])
        view.ingest(shared.snapshot(), 21_000_000_000)
        self.assertIn('THREE', view.calibration_failure)
        self.assertIn('CALIBRATION FAILED', view.unavailable_reason(['OK', 'OK'], True))

    def test_fresh_but_sparse_mocap_explains_rejected_pairs(self):
        view, shared = self.make_view()
        view.matrix, view.offset = np.eye(3), np.zeros(3)
        shared.set_tracker_pose('/body', 1_000_000_000, [1, 1, 1], [0, 0, 0, 1])
        shared.set_tracker_pose('/body', 1_200_000_000, [1, 1, 1], [0, 0, 0, 1])
        shared.set_tracker_pose('quest', 1_100_000_000, [1, 1, 1], [0, 0, 0, 1])
        view.render(1_200_000_000)
        self.assertEqual(view.paired_count, 0)
        self.assertIn('mocap gap 0.200s', view.unavailable_reason(['OK', 'OK'], False))
        self.assertIn('--max-gap', view.pairing_note)
        self.assertEqual(synchronization_issue([], 0, .1), 'no mocap poses')

    def test_waits_indefinitely_by_default_with_optional_deadline(self):
        self.assertIsNone(parse_args(['--mocap-topic', '/body']).startup_timeout_s)
        self.assertEqual(parse_args(['--mocap-topic', '/body', '--startup-timeout-s', '10']).startup_timeout_s, 10.)

    def test_alignment_handles_handedness_without_fitting_scale(self):
        q = np.random.default_rng(0).normal(size=(200, 3)) * .1
        transform = np.array([[0., 0., -1.], [1., 0., 0.], [0., 1., 0.]])
        m = q @ transform.T + [.4, -.2, .7]
        matrix, offset, rms = fit_alignment(q, m)
        np.testing.assert_allclose(matrix, transform, atol=1e-12)
        np.testing.assert_allclose(offset, [.4, -.2, .7], atol=1e-12)
        self.assertLess(rms, 1e-12)
        matrix, _, rms = fit_alignment(q, m * 1.1)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        self.assertGreater(rms, .01)

    def test_calibration_rejects_insufficient_or_planar_motion(self):
        q = np.random.default_rng(2).normal(size=(100, 3)) * .1
        q[:, 2] = 0
        with self.assertRaisesRegex(ValueError, 'THREE'):
            fit_alignment(q, q)
        with self.assertRaisesRegex(ValueError, '20'):
            fit_alignment(q[:10], q[:10])

    def test_interpolation_does_not_bridge_loss_gap_or_extrapolate(self):
        a, b = sample(1, [0, 0, 0]), sample(1.02, [.02, 0, 0])
        np.testing.assert_allclose(synchronized_position([a, b], 1_010_000_000, .1, [0, 0, 0]), [.01, 0, 0])
        self.assertIsNone(synchronized_position([a, b], 900_000_000, .1, [0, 0, 0]))
        self.assertIsNone(synchronized_position([a, b], 1_030_000_000, .1, [0, 0, 0]))
        self.assertIsNone(synchronized_position([a, b], 1_010_000_000, .01, [0, 0, 0]))
        self.assertIsNone(synchronized_position([a, sample(1.02, [.02, 0, 0], True)], 1_010_000_000, .1, [0, 0, 0]))
        np.testing.assert_allclose(synchronized_position([a, b], a.monotonic_ns, .1, [.1, .2, .3]), [.1, .2, .3])

    def test_frozen_requires_other_stream_motion_and_full_window(self):
        still = [(int(t * 1e9), [0, 0, 0], [0, 0, 0]) for t in np.linspace(1, 1.5, 51)]
        self.assertEqual(frozen_status(still, .5, .0002, .005), '')
        moving = [(t, q, [i * .001, 0, 0]) for i, (t, q, _) in enumerate(still)]
        self.assertIn('QUEST FROZEN', frozen_status(moving, .5, .0002, .005))
        self.assertEqual(frozen_status(moving[:10], .5, .0002, .005), '')
        self.assertIn('MOCAP FROZEN', frozen_status([(t, m, q) for t, q, m in moving], .5, .0002, .005))

    def test_missing_and_lost_signals(self):
        self.assertEqual(signal_status(None, 0, .5), 'NO DATA')
        self.assertIn('NO SIGNAL', signal_status(sample(1, [1, 0, 0]), 2_000_000_000, .5))
        self.assertIn('OBJECT LOST', signal_status(sample(1, [0, 0, 0], True), 1_000_000_000, .5))

    def test_worker_history_catches_error_between_ui_draws_and_renders(self):
        import cv2
        shared = PoseHistory()
        args = parse_args(['--mocap-topic', '/body'])
        with patch.object(cv2, 'namedWindow'), patch.object(cv2, 'setMouseCallback'):
            view = AccuracyView(shared, args)
        view.start_ns = 1_000_000_000
        view.matrix, view.offset = np.eye(3), np.zeros(3)
        for i, x in enumerate((0., .025, 0.)):
            shared.set_tracker_pose('quest', 1_000_000_000 + i * 10_000_000, [x, 0, 0], [0, 0, 0, 1])
            shared.set_tracker_pose('/body', 1_000_000_000 + i * 10_000_000, [0, 0, 0], [0, 0, 0, 1])
        image = view.render(1_020_000_000)
        self.assertEqual(view.count, 3)
        self.assertEqual(view.exceedances, 1)
        self.assertAlmostEqual(view.maximum, .025)
        self.assertIn("t=0.01s: 25.0 mm", view.last_exceedance)
        self.assertEqual(image.shape, (850, 1200, 3))
        self.assertTrue(np.any(np.all(image == (60, 60, 255), axis=2)))
        view.render(1_020_000_000)
        self.assertEqual(view.count, 3)  # No double counting when redrawing.
        view.render(2_020_000_000)
        self.assertIn('NO SIGNAL', view.last_event)


if __name__ == '__main__':
    unittest.main()
