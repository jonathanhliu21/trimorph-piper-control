"""Hardware-free regression checks for the passive Piper/Quest accuracy viewer."""
import unittest
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from data_collection.collection.shared_state import TrackerSample
from data_collection.collection.mocap_ik_monitor import MocapIKChecker
from scripts.quest.visualize_quest_accuracy_piper import (
    Comparison, PIPER, PiperHistory, PiperView, fit_capture, parse_args, run_piper, validate_calibration,
)
from data_collection.collection.quest_worker import QUEST_TRACKER_NAME


def sample(t, p, r=None, lost=False):
    r = Rotation.identity() if r is None else r
    return TrackerSample(round(t * 1e9), tuple(p), tuple(r.as_quat()), lost, round(t * 1000))


def identity():
    return {'linear_matrix': np.eye(3).tolist(), 'orientation': {'rotation_matrix': np.eye(3).tolist()}}


class PiperAccuracyTests(unittest.TestCase):
    def test_jump_markers_preserve_out_and_back_but_ignore_rotating_offset_and_gaps(self):
        cal = identity()
        cal['tracked_point_offset'] = {'frame': 'piper_flange', 'vector': [.1, 0, 0]}
        c = Comparison(parse_args([]), cal)
        times = (1., 1.01, 1.02, 1.03, 2.)
        positions = ([0, 0, 0], [.06, 0, 0], [0, 0, 0], [0, 0, 0], [.2, 0, 0])
        rotations = [Rotation.identity()] * 3 + [Rotation.from_euler('z', 90, degrees=True)] * 2
        h = {PIPER: [sample(t, [0, 0, 0], r) for t, r in zip(times, rotations)],
             QUEST_TRACKER_NAME: [sample(t, p, r) for t, p, r in zip(times, positions, rotations)]}
        c.ingest(h, 2_000_000_000)
        self.assertEqual([stamp for stamp, _ in c.jumps], [1_010_000_000, 1_020_000_000])
        np.testing.assert_allclose(c.jumps[0][1], [.06, 0, 0], atol=1e-10)
        c.ingest(h, 2_000_000_000)
        self.assertEqual(len(c.jumps), 2)
        c.clear()
        self.assertEqual(len(c.jumps), 0)

    def test_passive_reader_ignores_cached_feedback_and_disconnects(self):
        shared = PiperHistory()
        api = Mock()
        api.connect.return_value = True
        api.robot.fk.return_value = [.3, 0., .2, 0., 0., 0.]
        api.get_state.return_value = {}
        feedback = SimpleNamespace(timestamp=1., joints=[0.] * 6)
        reads = 0

        def read():
            nonlocal reads
            reads += 1
            if reads == 3:
                shared.shutdown.set()
            return feedback

        api.get_joint_state.side_effect = read
        with patch('scripts.piper_api.piper_api.PiperAPI', return_value=api) as factory:
            run_piper(parse_args([]), shared, init_done=threading.Event())
        self.assertEqual(len(shared.snapshot()[PIPER]), 1)
        self.assertIsNone(factory.call_args.kwargs['installation_pos'])
        self.assertIsNone(factory.call_args.kwargs['payload'])
        self.assertEqual([call[0] for call in api.method_calls],
                         ['connect', 'get_joint_state', 'robot.fk', 'get_state',
                          'get_joint_state', 'get_joint_state', 'disconnect'])

    def test_latency_sign_and_interpolated_orientation(self):
        args = parse_args([])
        cal = dict(identity(), latency_s=.1)
        c = Comparison(args, cal)
        piper = [sample(1., [0, 0, 0]), sample(1.02, [.02, 0, 0], Rotation.from_euler('z', .2))]
        quest = [sample(1.10, [1, 0, 0]), sample(1.11, [1.01, 0, 0], Rotation.from_euler('z', .1))]
        c.ingest({PIPER: piper, QUEST_TRACKER_NAME: quest}, 1_120_000_000)
        self.assertEqual(c.count, 2)
        np.testing.assert_allclose(c.latest[-1], [0, 0], atol=1e-10)
        np.testing.assert_allclose(c.latest[3], [.01, 0, 0], atol=1e-10)

    def test_waits_for_bracket_and_never_counts_duplicate_samples(self):
        c = Comparison(parse_args([]), identity())
        h = {PIPER: [sample(1, [0, 0, 0])], QUEST_TRACKER_NAME: [sample(1.01, [0, 0, 0])]}
        self.assertEqual(c.ingest(h, 1_020_000_000), [])
        h[PIPER].append(sample(1.02, [0, 0, 0]))
        c.ingest(h, 1_030_000_000)
        c.ingest(h, 1_040_000_000)
        self.assertEqual(c.count, 1)

    def test_rejects_gaps_lost_and_invalid_quaternions(self):
        for lost, end, quat in ((True, 1.02, None), (False, 1.2, None), (False, 1.02, (0, 0, 0, 0))):
            c = Comparison(parse_args([]), identity())
            q = sample(1.01, [0, 0, 0])
            if quat is not None:
                q = TrackerSample(q.monotonic_ns, q.position, quat, False, 1)
            c.ingest({PIPER: [sample(1, [0, 0, 0], lost=lost), sample(end, [0, 0, 0])],
                      QUEST_TRACKER_NAME: [q]}, 1_300_000_000)
            self.assertEqual(c.count, 0)

    def test_mapping_preserves_flange_offset_under_rotation(self):
        cal = identity()
        cal['tracked_point_offset'] = {'frame': 'piper_flange', 'vector': [.1, 0, 0]}
        c = Comparison(parse_args([]), cal)
        r = Rotation.from_euler('z', 90, degrees=True)
        h = {PIPER: [sample(1, [.1, 0, 0]), sample(2, [0, .1, 0], r)],
             QUEST_TRACKER_NAME: [sample(1, [0, 0, 0]), sample(2, [0, 0, 0], r)]}
        c.ingest(h, 2_000_000_000)
        np.testing.assert_allclose(c.latest[-1], [0, 0], atol=1e-10)
        c.reanchor()
        self.assertEqual(c.count, 0)
        self.assertIsNone(c.anchor)
        self.assertEqual(c.generation, 1)

    def test_live_fit_recovers_reflection_and_offset_and_predicts_unseen_poses(self):
        rng = np.random.default_rng(42)
        a = Rotation.from_euler('xyz', [.2, -.4, .8]).as_matrix() @ np.diag([-1, 1, 1])
        b = Rotation.from_euler('xyz', [.3, -.2, .6])
        tool = Rotation.from_euler('xyz', [.2, -.1, .3])
        lever = np.array([.08, -.03, .1])
        q = rng.normal(size=(160, 3)) * .15
        rq = Rotation.from_rotvec(rng.normal(size=(160, 3)) * .6)
        rp = b * rq * tool
        p = q @ a.T + rp.apply(lever) + [.2, .1, .3]
        pairs = list(zip(q[:120], rq.as_quat()[:120], p[:120], rp.as_matrix()[:120]))
        cal = fit_capture(pairs, .02)
        np.testing.assert_allclose(cal['linear_matrix'], a, atol=1e-7)
        np.testing.assert_allclose(cal['tracked_point_offset']['vector'], lever, atol=1e-7)
        c = Comparison(parse_args(['--latency-s', '0']), cal)
        h = {PIPER: [sample(i, p[i], rp[i]) for i in range(120, 160)],
             QUEST_TRACKER_NAME: [sample(i, q[i], rq[i]) for i in range(120, 160)]}
        c.ingest(h, 160_000_000_000)
        self.assertEqual(c.count, 40)
        np.testing.assert_allclose(c.maximum, [0, 0], atol=1e-5)

    def test_live_calibration_excludes_measurements_and_rejects_degenerate_motion(self):
        c = Comparison(parse_args([]), identity())
        c.start_capture(1_000_000_000)
        c.ingest({PIPER: [sample(2, [0, 0, 0])], QUEST_TRACKER_NAME: [sample(2, [0, 0, 0])]}, 2_000_000_000)
        self.assertEqual(c.count, 0)
        self.assertEqual(len(c.capture), 1)
        with self.assertRaises(ValueError):
            fit_capture(c.capture * 30, 0.)

    def test_validation_rejects_bad_calibration_and_arguments(self):
        for cal in ({}, dict(identity(), latency_s=float('nan')), dict(identity(), tracker='mocap')):
            with self.assertRaises(ValueError):
                validate_calibration(cal)
        with self.assertRaises(SystemExit):
            parse_args(['--max-gap', '0'])
        with self.assertRaises(SystemExit):
            parse_args(['--quest-port', '0'])

    def test_collection_ik_accepts_already_mapped_piper_pose(self):
        checker = MocapIKChecker(None, [0, 0, 0, 0, 30, 0],
                                 model=(lambda q: np.array(q), np.tile([-3., 3.], (6, 1))))
        target = sample(1, [.2, .1, .3], Rotation.from_euler('y', .5))
        result = checker.check(target)
        self.assertTrue(result.ok, result.detail)
        np.testing.assert_allclose(np.radians(result.joints_deg), [.2, .1, .3, 0, .5, 0], atol=1e-6)

    def test_renderer_runs_headless_for_waiting_valid_stale_and_ik_warning(self):
        from data_collection.collection.mocap_ik_monitor import IKResult
        args, shared = parse_args([]), PiperHistory()
        c = Comparison(args, identity())
        with patch.object(cv2, 'namedWindow'), patch.object(cv2, 'setMouseCallback'):
            view = PiperView(shared, args, c)
        self.assertEqual(view.draw({}, 1_000_000_000, None).shape, (900, 1280, 3))
        h = {PIPER: [sample(1, [.3, 0, .2])], QUEST_TRACKER_NAME: [sample(1, [.1, 0, .1])]}
        c.ingest(h, 1_000_000_000)
        result = (c.generation, IKResult(1_000_000_000, False, 'Wrist singularity margin violated', (0, 0, 0, 0, 0, 0)))
        for now in (1_000_000_000, 3_000_000_000):
            frame = view.draw(h, now, result)
            self.assertTrue(np.any(frame != 28))


if __name__ == '__main__':
    unittest.main()
