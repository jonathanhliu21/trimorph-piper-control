import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from data_collection import postprocess_dataset as post
from data_collection.replay_episode import (
    anchor_relative_piper_stream, load_piper_stream, resolve_piper_source, main as replay_main,
)


class PostprocessTests(unittest.TestCase):
    def write_csv(self, path, fields, rows):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def fixture(self, root, *, no_camera=True, start_pose=None):
        episode = root / 'raw' / 'episode_000000'
        episode.mkdir(parents=True)
        origin = 1_000_000_000
        stamps = [origin + i * 20_000_000 for i in range(4)]
        (episode / 'metadata.json').write_text(json.dumps(dict(status='complete', started_monotonic_ns=origin)))
        fields = [f'pos_{i}' for i in range(13)]
        self.write_csv(episode / 'leader_hand.csv', ['monotonic_ns', *fields],
                       [dict(monotonic_ns=ns, **{key: i + k / 10 for k, key in enumerate(fields)})
                        for i, ns in enumerate(stamps)])
        qxyz = np.array([[2., 3., 4.], [2.1, 3., 4.], [2.1, 3.2, 4.], [2.1, 3.2, 4.3]])
        rq = Rotation.from_euler('xyz', [[.1, .2, .3], [.2, .1, .3], [.2, .3, .1], [.4, .1, .2]])
        self.write_csv(episode / 'quest.csv', ['monotonic_ns', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'],
                       [dict(monotonic_ns=ns, **dict(zip(['px','py','pz','qx','qy','qz','qw'], [*p, *q])))
                        for ns, p, q in zip(stamps, qxyz, rq.as_quat())])
        (episode / 'piper.csv').write_text('')
        a = np.array([[0., -2., 0.], [2., 0., 0.], [0., 0., 2.]])
        b = Rotation.from_euler('xyz', [.4, -.3, .2]).as_matrix()
        calibration = root / 'calibration.json'
        calibration.write_text(json.dumps(dict(direction='quest_to_piper', linear_matrix=a.tolist(),
                                               orientation=dict(rotation_matrix=b.tolist()))))
        config = post.PostprocessConfig(root, root / 'raw', False, {'complete'}, .01, 1,
                                        calibration, no_camera, start_pose)
        raw = post.base.discover_raw_episodes(config)[0]
        return config, raw, qxyz, rq, a, b

    def test_quest_no_camera_outputs_and_replay_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, raw, qxyz, rq, a, b = self.fixture(root)
            with patch.object(post.base, 'import_cv2', side_effect=AssertionError('Camera accessed')):
                summary = post.process_episode(config, raw)
            self.assertEqual(raw.piper_path.read_text(), '')
            self.assertIsNone(summary['video'])
            self.assertFalse((root / 'camera').exists())
            hand = np.load(summary['hand'])
            eef = np.load(summary['eef'])
            self.assertEqual(hand.shape, (4, 13))
            self.assertEqual(eef.shape, (4, 6))
            self.assertEqual(eef.dtype, np.float32)
            expected_xyz = (qxyz - qxyz[0]) @ a.T
            expected_r = b @ (rq.as_matrix() @ rq.as_matrix()[0].T) @ b.T
            np.testing.assert_allclose(eef[:, :3], expected_xyz, atol=1e-7)
            np.testing.assert_allclose(Rotation.from_rotvec(eef[:, 3:]).as_matrix(), expected_r, atol=1e-7)
            path, frame = resolve_piper_source(raw.path)
            self.assertEqual(frame, 'relative_to_episode_start')
            self.assertEqual(path, Path(summary['piper_csv']))
            stream = load_piper_stream(path)
            start = np.array([.4, -.1, .3, .3, .5, -.4])
            times, poses = anchor_relative_piper_stream(stream, start)
            np.testing.assert_allclose(poses[:, :3], start[:3] + expected_xyz, atol=1e-10)
            np.testing.assert_allclose(Rotation.from_euler('xyz', poses[:, 3:]).as_matrix(),
                                       expected_r @ Rotation.from_euler('xyz', start[3:]).as_matrix(), atol=1e-10)
            np.testing.assert_allclose(poses[0], start, atol=1e-10)
            with np.load(summary['sync']) as sync:
                self.assertNotIn('frame_idx', sync)
                np.testing.assert_array_equal(sync['encoder_idx'], np.arange(4))
                np.testing.assert_array_equal(sync['pose_idx'], np.arange(4))
            self.assertEqual(resolve_piper_source(raw.path, read_piper_raw=True), (raw.piper_path, 'piper_base'))
            with self.assertRaises(FileExistsError):
                post.process_episode(config, raw)

    def test_camera_copy_trim_and_reference_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            config, raw, *_ = self.fixture(Path(directory), no_camera=False, start_pose=(.4, 0., .3, .2, .1, 0.))
            raw.camera_path.write_bytes(b'test video')
            self.write_csv(raw.camera_timestamps_path, ['frame_idx', 'monotonic_ns'],
                           [dict(frame_idx=i, monotonic_ns=1_000_000_000+i*20_000_000) for i in range(4)])
            cv2 = MagicMock()
            cv2.VideoCapture.return_value.get.return_value = 4
            with patch.object(post.base, 'import_cv2', return_value=cv2):
                summary = post.process_episode(config, raw)
            self.assertEqual(Path(summary['video']).read_bytes(), b'test video')
            with np.load(summary['sync']) as sync:
                np.testing.assert_array_equal(sync['frame_idx'], [1, 2])
                np.testing.assert_array_equal(sync['encoder_idx'], [1, 2])
                np.testing.assert_array_equal(sync['pose_idx'], [1, 2])
            self.assertEqual(resolve_piper_source(raw.path)[1], 'piper_base')
            first = load_piper_stream(Path(summary['piper_csv']))[1][0]
            np.testing.assert_allclose(first, config.piper_start_pose, atol=1e-10)

    def test_offset_conversion_requires_start_pose_and_applies_rotating_correction(self):
        for start in (None, (.4, -.1, .3, .2, -.4, .1)):
            with self.subTest(start=start), tempfile.TemporaryDirectory() as directory:
                config, raw, qxyz, rq, a, b = self.fixture(Path(directory), start_pose=start)
                calibration = json.loads(config.calibration_json.read_text())
                offset = np.array([-.09, .03, -.04])
                calibration['tracked_point_offset'] = dict(vector=offset.tolist(), frame='piper_flange')
                config.calibration_json.write_text(json.dumps(calibration))
                if start is None:
                    with self.assertRaisesRegex(ValueError, '--piper-start-pose'):
                        post.process_episode(config, raw)
                    self.assertFalse((config.dataset_root / 'processed').exists())
                    continue
                summary = post.process_episode(config, raw)
                mapped_r = b @ (rq.as_matrix() @ rq.as_matrix()[0].T) @ b.T
                rp0 = Rotation.from_euler('xyz', start[3:]).as_matrix()
                expected = start[:3] + (qxyz - qxyz[0]) @ a.T + (mapped_r @ rp0 - rp0) @ offset
                poses = load_piper_stream(Path(summary['piper_csv']))[1]
                np.testing.assert_allclose(poses[:, :3], expected, atol=1e-10)
                self.assertEqual(resolve_piper_source(raw.path)[1], 'piper_base')
                for invalid in (dict(frame='quest', vector=offset.tolist()),
                                dict(frame='piper_flange', vector=[0., 1.]),
                                dict(frame='piper_flange', vector=[0., float('nan'), 0.])):
                    calibration['tracked_point_offset'] = invalid
                    with self.assertRaisesRegex(ValueError, 'tracked_point_offset'):
                        post.validate_calibration(calibration)

    def test_start_pose_recorded_at_collection_is_used_when_not_given(self):
        recorded = [.4, -.1, .3, .2, -.4, .1]
        with tempfile.TemporaryDirectory() as directory:
            config, raw, *_ = self.fixture(Path(directory))
            metadata = json.loads(raw.metadata_path.read_text())
            raw.metadata_path.write_text(json.dumps(dict(metadata, piper=dict(start_pose=recorded))))
            summary = post.process_episode(config, raw)
            np.testing.assert_allclose(load_piper_stream(Path(summary['piper_csv']))[1][0], recorded, atol=1e-10)
            self.assertEqual(resolve_piper_source(raw.path)[1], 'piper_base')
            processing = json.loads(Path(summary['metadata']).read_text())
            self.assertEqual(processing['piper_start_pose'], recorded)

    def test_absolute_mode_keeps_controller_position_and_rejects_far_starts(self):
        start = (.4, -.1, .3, .2, -.4, .1)
        with tempfile.TemporaryDirectory() as directory:
            config, raw, qxyz, rq, a, b = self.fixture(Path(directory), start_pose=start)
            calibration = json.loads(config.calibration_json.read_text())
            offset = np.array([-.09, .03, -.04])
            origin = np.array(start[:3]) - qxyz[0] @ a.T + np.array([.02, -.01, .03])  # hand 20/-10/30 mm off
            calibration.update(tracked_point_offset=dict(vector=offset.tolist(), frame='piper_flange'),
                               position_offset=origin.tolist())
            config.calibration_json.write_text(json.dumps(calibration))
            config = replace(config, absolute=True)
            summary = post.process_episode(config, raw)
            rp0 = Rotation.from_euler('xyz', start[3:]).as_matrix()
            mapped_r = b @ (rq.as_matrix() @ rq.as_matrix()[0].T) @ b.T @ rp0
            expected = qxyz @ a.T + mapped_r @ offset + origin
            poses = load_piper_stream(Path(summary['piper_csv']))[1]
            np.testing.assert_allclose(poses[:, :3], expected, atol=1e-10)
            np.testing.assert_allclose(Rotation.from_euler('xyz', poses[0, 3:]).as_matrix(), rp0, atol=1e-10)
            processing = json.loads(Path(summary['metadata']).read_text())
            self.assertEqual(processing['position_mapping'], 'absolute_calibration_origin')
            np.testing.assert_allclose(processing['first_pose_offset_from_start_m'],
                                       expected[0] - start[:3], atol=1e-10)

            far = replace(config, max_start_offset_mm=10.0, overwrite_processed_files=True)
            with self.assertRaisesRegex(ValueError, 'Quest tracking frame'):
                post.process_episode(far, raw)

    def test_mocap_source_replays_the_flange_pose_of_the_calibrated_marker_plate(self):
        start = (.4, -.1, .3, .2, -.4, .1)
        w = Rotation.from_euler('zyx', [90, 2, -1], degrees=True)  # Vicon world -> Piper base
        t_w = np.array([.1, -1.3, .8])
        x = Rotation.from_euler('xyz', [180, 5, -90], degrees=True)  # flange in plate frame
        t_x = np.array([-.069, .002, -.100])
        # Flange-equivalent poses in Vicon world: a translation, then a 30 deg turn about the flange point.
        f_world = Rotation.from_euler('xyz', [[0, 0, 0], [0, 0, 0], [30, 0, 0], [30, 10, 0]], degrees=True)
        f_pos = np.array([[1.2, -.4, 1.25], [1.21, -.4, 1.25], [1.21, -.4, 1.25], [1.21, -.38, 1.24]])
        plate_r = f_world * x.inv()
        plate_pos = f_pos - plate_r.apply(t_x)
        with tempfile.TemporaryDirectory() as directory:
            config, raw, *_ = self.fixture(Path(directory), start_pose=start)
            calibration = json.loads(config.calibration_json.read_text())
            calibration['vicon_reference'] = {'run': dict(
                vicon_to_piper_rotation=w.as_matrix().tolist(), vicon_to_piper_translation_m=t_w.tolist(),
                flange_in_vicon_body_rotation=x.as_matrix().tolist(), flange_in_vicon_body_translation_m=t_x.tolist())}
            config.calibration_json.write_text(json.dumps(calibration))
            pose = 'vicon_receiver/msg/Position'
            rows = [dict(monotonic_ns=1_000_000_000 + i * 20_000_000, topic='/vicon/Yori/Yori', message_type=pose,
                         **dict(zip(['x_trans', 'y_trans', 'z_trans', 'x_rot', 'y_rot', 'z_rot', 'w'], [*p, *q])))
                    for i, (p, q) in enumerate(zip(1000 * plate_pos, plate_r.as_quat()))]
            rows.insert(2, dict(rows[1], monotonic_ns=1_030_000_000, x_trans=0, y_trans=0, z_trans=0,
                                x_rot=0, y_rot=0, z_rot=0, w=0))  # lost object
            rows.append(dict(rows[0], topic='/vicon/Hand/Hand'))
            rows.append(dict(monotonic_ns=1_000_000_000, topic='/vicon/markers',
                             message_type='vicon_receiver/msg/MarkerPositions', x_trans=1, y_trans=2, z_trans=3))
            fields = ['monotonic_ns', 'topic', 'message_type', 'x_trans', 'y_trans', 'z_trans', 'x_rot', 'y_rot', 'z_rot', 'w']
            self.write_csv(raw.path / 'mocap.csv', fields, rows)

            mocap = replace(config, pose_source='mocap', mocap_smooth_s=0)  # exact geometry, no filtering
            with self.assertRaisesRegex(ValueError, '--mocap-topic'):
                post.process_episode(mocap, raw)
            mocap = replace(mocap, mocap_topic='/vicon/Yori/Yori')
            summary = post.process_episode(mocap, raw)
            poses = load_piper_stream(Path(summary['piper_csv']))[1]
            flange_base = f_pos @ w.as_matrix().T + t_w
            # Relative: the plate-to-flange offset is removed, so turning about the flange does not move it.
            np.testing.assert_allclose(poses[:, :3], np.array(start[:3]) + flange_base - flange_base[0], atol=1e-6)
            np.testing.assert_allclose(poses[1, :3], poses[2, :3], atol=1e-6)
            r_base = w * f_world * Rotation.from_matrix(np.eye(3))
            expected_r = (r_base * r_base[0].inv()).as_matrix() @ Rotation.from_euler('xyz', start[3:]).as_matrix()
            np.testing.assert_allclose(Rotation.from_euler('xyz', poses[:, 3:]).as_matrix(), expected_r, atol=1e-6)
            processing = json.loads(Path(summary['metadata']).read_text())
            self.assertEqual((processing['source'], processing['mocap_topic']), ('mocap.csv', '/vicon/Yori/Yori'))

            absolute = replace(mocap, absolute=True, overwrite_processed_files=True,
                               max_start_offset_mm=1e6, max_start_rotation_deg=180)
            summary = post.process_episode(absolute, raw)
            poses = load_piper_stream(Path(summary['piper_csv']))[1]
            np.testing.assert_allclose(poses[:, :3], flange_base, atol=1e-6)
            np.testing.assert_allclose(Rotation.from_euler('xyz', poses[:, 3:]).as_matrix(),
                                       (w * f_world).as_matrix(), atol=1e-6)
            with self.assertRaisesRegex(ValueError, 'mount near the arm'):
                post.process_episode(replace(absolute, max_start_offset_mm=150.0), raw)

            del calibration['vicon_reference']
            config.calibration_json.write_text(json.dumps(calibration))
            with self.assertRaisesRegex(ValueError, 'reference mocap'):
                post.process_episode(absolute, raw)

    def test_vicon_smoothing_removes_orientation_flicks_and_keeps_slow_motion(self):
        n = 300
        t = np.arange(n) / 300.0
        positions = np.column_stack([.1 * np.sin(2 * np.pi * t), np.zeros(n), np.zeros(n)])
        rotations = Rotation.from_euler('z', (40 * np.sin(2 * np.pi * t))[:, None], degrees=True)
        quats = rotations.as_quat()
        quats[[50, 120, 121]] = Rotation.from_euler('x', 16, degrees=True).as_quat()  # marker flicks
        quats[200:] *= -1  # equivalent sign flip must not be averaged away
        smoothed_p, smoothed_q = post.smooth_poses(positions, quats, 1 / 300.0, 0.02)
        error = np.degrees((Rotation.from_quat(smoothed_q) * rotations.inv()).magnitude())
        self.assertLess(error[20:-20].max(), 1.0)  # ends: the test signal is already moving at t=0
        np.testing.assert_allclose(smoothed_p[20:-20], positions[20:-20], atol=2e-3)
        raw_p, raw_q = post.smooth_poses(positions, quats, 1 / 300.0, 0)
        np.testing.assert_array_equal(raw_q, quats)

    def test_vicon_frame_clock_undoes_bursty_delivery(self):
        frames = np.arange(100, 400, dtype=np.int64)
        true_ns = 5_000_000_000 + (frames - 100) * 3_333_333
        latency = np.where((frames // 30) % 2 == 0, 2_000_000, 0)  # alternate on-time and bursty arrivals
        receipt = true_ns + latency + (frames % 30) * 0  # bursts: same latency, then catch up
        receipt[30:60] = receipt[60]  # 30 frames delivered at once
        clock = post.frame_clock(frames, receipt)
        steps = np.diff(clock)
        self.assertLess(np.ptp(steps), 10_000)  # evenly spaced to within 10 us
        self.assertAlmostEqual(np.median(steps) / 1e6, 3.33, delta=0.2)
        self.assertLessEqual(np.percentile(receipt - clock, 5), 1)  # anchored at the earliest arrivals

    def test_replay_default_anchors_processed_and_raw_flag_preserves_raw_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, raw, *_ = self.fixture(root)
            summary = post.process_episode(config, raw)
            raw.piper_path.write_text(Path(summary['piper_csv']).read_text())
            measured = [.4, -.1, .3, .2, -.1, .3]
            for raw_mode in (False, True):
                api = MagicMock()
                api.is_ok.return_value = True
                api.enable.return_value = True
                api.get_eef_pose.return_value = measured
                argv = ['replay_episode.py', '0', '--dataset-root', str(root), '--arm-only', '--no-preposition']
                if raw_mode:
                    argv.append('--read-piper-raw')
                with (patch('sys.argv', argv),
                      patch('scripts.piper_api.piper_api.PiperAPI', return_value=api),
                      patch('data_collection.replay_episode.replay_piper_commands') as commands):
                    self.assertEqual(replay_main(), 0)
                replayed = commands.call_args.args[1][1]
                if raw_mode:
                    api.get_eef_pose.assert_not_called()
                    np.testing.assert_allclose(replayed[0], np.zeros(6), atol=1e-10)
                else:
                    api.get_eef_pose.assert_called_once()
                    np.testing.assert_allclose(replayed[0], measured, atol=1e-10)
                api.disconnect.assert_called_once()

    def test_missing_processed_data_has_no_implicit_raw_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, raw, *_ = self.fixture(root)
            with self.assertRaisesRegex(FileNotFoundError, 'read-piper-raw'):
                resolve_piper_source(raw.path)
            self.assertEqual(resolve_piper_source(raw.path, read_piper_raw=True)[0], raw.piper_path)

    def test_invalid_calibration_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            config, raw, *_ = self.fixture(Path(directory))
            config.calibration_json.write_text(json.dumps(dict(direction='quest_to_piper', linear_matrix=np.eye(3).tolist())))
            with self.assertRaisesRegex(ValueError, 'orientation.rotation_matrix'):
                post.process_episode(config, raw)
            self.assertFalse((config.dataset_root / 'processed').exists())


if __name__ == '__main__':
    unittest.main()
