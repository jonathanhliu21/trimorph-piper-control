import unittest
from unittest.mock import patch

import numpy as np

from scripts.quest.visualize_quest import LatestPose, Playback, QuestView, parse_args, project_unity, rotation_matrix
from data_collection.collection.shared_state import SharedState


class LiveJumpTests(unittest.TestCase):
    def append(self, poses, index, position):
        poses.append_quest(dict(px=position, py=0., pz=0., qx=0., qy=0., qz=0., qw=1.,
                                monotonic_ns=1_000_000_000 + index * 20_000_000, seq=index))

    def test_jump_out_and_back_between_draws_is_counted(self):
        poses = LatestPose(jump_threshold=.05)
        self.append(poses, 0, 0.)
        self.append(poses, 1, 1.)
        self.append(poses, 2, 0.)
        # The latest displayed pose is unchanged, but both received jumps matter.
        row, count, jump = poses.snapshot()
        self.assertEqual(count, 2)
        self.assertEqual(row['px'], 0.)
        self.assertEqual(jump.start, (1., 0., 0.))
        self.assertEqual(jump.end, (0., 0., 0.))
        self.assertEqual(jump.distance, 1.)
        self.assertAlmostEqual(jump.interval_s, .02)
        for _ in range(3):
            self.assertEqual(poses.snapshot()[1], 2)
        row['px'] = 999.
        self.assertEqual(poses.latest()['px'], 0.)

    def test_threshold_and_ordinary_motion(self):
        poses = LatestPose(jump_threshold=.05)
        self.assertEqual(poses.snapshot(), (None, 0, None))
        self.append(poses, 0, 0.)
        self.append(poses, 1, .05)
        self.append(poses, 2, .051)
        self.assertEqual(poses.snapshot()[1], 0)
        self.append(poses, 3, .2)
        self.assertEqual(poses.snapshot()[1], 1)
        self.append(poses, 4, .201)
        self.assertEqual(poses.snapshot()[1], 1)


class OpenCVViewTests(unittest.TestCase):
    def test_unity_y_moves_up_without_sideways_motion_at_any_camera_yaw(self):
        for yaw in (-1., 0., .8):
            points = project_unity([[0, 0, 0], [0, .1, 0]], yaw, .4, 1000, (600, 415))
            self.assertEqual(points[0, 0], points[1, 0])
            self.assertLess(points[1, 1], points[0, 1])
        # With a front-facing camera, Unity +X is right and +Y is up.
        points = project_unity(np.eye(3), 0., 0., 100, (0, 0))
        np.testing.assert_array_equal(points, [[100, 0], [0, -100], [0, 0]])

    def test_unity_y_rotation_takes_local_forward_to_world_right(self):
        row = dict(qx=0., qy=2 ** -.5, qz=0., qw=2 ** -.5)
        forward = rotation_matrix(row) @ [0., 0., 1.]
        np.testing.assert_allclose(forward, [1., 0., 0.], atol=1e-12)
        point = project_unity(forward, 0., 0., 100, (0, 0))
        np.testing.assert_array_equal(point, [[100, 0]])

    def make_view(self, playback=None):
        import cv2
        with patch.object(cv2, 'namedWindow'), patch.object(cv2, 'setMouseCallback'):
            return QuestView(SharedState(), parse_args([]), playback)

    def row(self, x=0., qz=0., qw=1.):
        return dict(px=x, py=0., pz=0., qx=0., qy=0., qz=qz, qw=qw,
                    seq=1, monotonic_ns=1_000_000_000)

    def test_live_render_shows_rotation_and_keeps_fixed_display_origin(self):
        view = self.make_view()
        a = self.row()
        b = self.row(qz=2 ** -.5, qw=2 ** -.5)
        image_a = view.draw(a, [a], 0, now_ns=1_000_000_000)
        image_b = view.draw(b, [b], 0, now_ns=1_000_000_000)
        self.assertEqual(image_a.shape, (900, 1200, 3))
        self.assertFalse(np.array_equal(image_a[220:600], image_b[220:600]))
        view.draw(self.row(.1), [self.row(.1)], 0, now_ns=1_000_000_000)
        np.testing.assert_array_equal(view.display_origin, [0, 0, 0])
        view.key(ord('r'))
        np.testing.assert_allclose(view.display_origin, [.1, 0, 0])
        # No data and invalid orientations still produce a usable window.
        view.draw(None, [], 0)
        view.draw(self.row(qw=0.), [], 0)

    def test_csv_clock_navigation_mouse_seek_and_render(self):
        times = np.array([0., .1, .3, .7])
        playback = Playback(times, np.array([1, 3]), 2.)
        playback.tick(playback.last_wall + .1)
        self.assertAlmostEqual(playback.value, .2)
        view = self.make_view(playback)
        view.key(65363)
        self.assertEqual(playback.index, 2)
        self.assertFalse(playback.playing)
        view.key(ord('k'))
        self.assertEqual(playback.index, 3)
        view.key(ord('j'))
        self.assertEqual(playback.index, 1)
        view._on_mouse(view.cv2.EVENT_LBUTTONDOWN, 600, 700, 0, None)
        self.assertAlmostEqual(playback.value, .35)
        self.assertFalse(playback.playing)
        rows = [self.row(x) for x in [0., .1, .11, .3]]
        positions = np.array([[r['px'], 0., 0.] for r in rows])
        view.timeline = times, positions, np.array([1, 3])
        image = view.draw(rows[playback.index], rows, 2)
        self.assertEqual(image.shape, (900, 1200, 3))
        view._on_mouse(view.cv2.EVENT_LBUTTONDOWN, 80, 830, 0, None)
        self.assertTrue(playback.playing)
        playback.tick(playback.last_wall + 10.)
        self.assertEqual(playback.value, .7)
        self.assertFalse(playback.playing)
        view.key(ord(' '))
        self.assertEqual(playback.value, 0.)
        self.assertTrue(playback.playing)


if __name__ == '__main__':
    unittest.main()
