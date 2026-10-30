import unittest

from scripts.quest.visualize_quest import LatestPose


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


if __name__ == '__main__':
    unittest.main()
