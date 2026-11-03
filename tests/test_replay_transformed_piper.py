import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from data_collection.replay_actions import Limits
from scripts.replay_corl_workshop import replay_transformed_piper as replay


class TransformedReplayTests(unittest.TestCase):
    def setUp(self):
        self.bounds = np.tile([-1., 1.], (6, 1))
        self.seed = np.array([0., 0., 0., 0., .4, 0.])
        self.limits = Limits(joint_margin_deg=3, max_joint_step_deg=2)

    @staticmethod
    def fk(q):
        return np.asarray(q)

    def poses(self, joints):
        return np.c_[joints[:, :3], Rotation.from_euler('xyz', joints[:, 3:]).as_rotvec()]

    def analyze(self, joints, **kwargs):
        return replay.analyze(np.arange(len(joints), dtype=float), self.poses(joints),
                              self.seed, self.bounds, kwargs.pop('fk', self.fk),
                              limits=self.limits, **kwargs)

    def test_reports_all_limits_and_jumps_at_original_rows(self):
        joints = np.tile(self.seed, (4, 1))
        joints[1, 0] = .98  # Inside hard bound, outside configured margin.
        joints[2:, 0] = 1.2
        joints[2:, 1] = -1.1
        result = self.analyze(joints)
        np.testing.assert_allclose(result['joints'], joints, atol=1e-7)
        violations = [(e['frame'], e['joint']) for e in result['events'] if e['kind'] == 'joint_limit']
        self.assertEqual(violations, [(2, 'J1'), (2, 'J2'), (3, 'J1'), (3, 'J2')])
        margins = [e for e in result['events'] if e['kind'] == 'joint_margin']
        self.assertEqual([(e['frame'], e['joint']) for e in margins], [(1, 'J1')])
        jump = next(e for e in result['events'] if e['kind'] == 'joint_jump')
        self.assertEqual((jump['previous_frame'], jump['frame'], jump['joint']), (0, 1, 'J1'))
        self.assertAlmostEqual(jump['jump_deg'], np.degrees(.98))
        self.assertIn('frame 0 -> 1', replay.describe(jump))

    def test_failed_fit_is_not_reported_as_joint_violation_or_jump(self):
        joints = np.tile(self.seed, (3, 1))
        joints[1, 0] = 2.

        def limited_fk(q):
            pose = q.copy()
            pose[0] = np.tanh(q[0])
            return pose

        result = self.analyze(joints, fk=limited_fk)
        self.assertTrue(np.isnan(result['joints'][1]).all())
        self.assertTrue(np.isfinite(result['joints'][2]).all())
        self.assertEqual([e['frame'] for e in result['events'] if e['kind'] == 'ik_failure'], [1])
        self.assertFalse(any(e['kind'] in ('joint_jump', 'joint_limit') for e in result['events']))

    def test_rotvec_wrap_does_not_create_false_rotation_jump(self):
        joints = np.tile(self.seed, (2, 1))
        joints[:, 3] = np.radians([179.9, 180.1])
        result = self.analyze(joints)
        self.assertFalse(any(e['kind'] == 'rotation_jump' for e in result['events']))

    def test_headless_dataset_export_uses_sync_and_needs_no_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'absolute_action_t').mkdir()
            (root/'syncs').mkdir()
            joints = np.tile(self.seed, (2, 1))
            joints[1, 0] = 1.1
            np.save(root/'absolute_action_t/000003.npy', self.poses(joints))
            np.savez(root/'syncs/000003.npz', pose_frame='piper_base', pose_monotonic_ns=[100, 100000100])
            with patch.object(replay, 'piper_model', return_value=(self.fk, self.bounds)), \
                    contextlib.redirect_stdout(io.StringIO()):
                replay.main([str(root/'absolute_action_t'), '--episode', '3', '--no-show',
                             '--start-joints-deg', *map(str, np.degrees(self.seed)),
                             '--output-dir', str(root/'output')])
            self.assertGreater((root/'output/000003.png').stat().st_size, 1000)
            with np.load(root/'output/000003.npz') as saved:
                np.testing.assert_allclose(saved['times'], [0, .1])
                np.testing.assert_allclose(saved['joints'], joints, atol=1e-7)
            with (root/'output/000003.csv').open() as stream:
                events = list(csv.DictReader(stream))
            self.assertTrue(any(e['kind'] == 'joint_limit' and e['frame'] == '1' for e in events))

    def test_single_frame_plot(self):
        import matplotlib
        matplotlib.use('Agg')
        result = self.analyze(self.seed[None, :])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'single.png'
            replay.visualize(result, self.bounds, self.limits, 'single', path, show=False)
            self.assertTrue(path.exists())


if __name__ == '__main__':
    unittest.main()
