"""Hardware-free checks of the accuracy viewer's measurement and rendering."""
import unittest
from unittest.mock import patch

import numpy as np

from scripts.quest.visualize_quest_accuracy import (
    AccuracyView, PoseHistory, fit_alignment, frozen_status, parse_args,
    signal_status, synchronized_position,
)
from data_collection.collection.shared_state import TrackerSample


def sample(t, p, lost=False):
    return TrackerSample(round(t * 1e9), tuple(p), (0., 0., 0., 1.), lost, 1)


class AccuracyTests(unittest.TestCase):
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
