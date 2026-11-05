"""Hardware-free coverage of the deployment coordinate and wire contracts."""
import contextlib
from dataclasses import dataclass
import io
import json
import socket
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from data_collection.collection import config
from inference import protocol
from inference.executor.executor import (
    FakeCamera, FakeRobot, MAX_EEF_JUMP_M, OakCamera, RobotIO, jitter_start_joints,
    parse_args, pose_to_piper, pose_to_policy, run,
)


@dataclass
class SimpleConfig:
    operating_mode: int = 5


class InferenceTests(unittest.TestCase):
    def setUp(self):
        # Start moves append to the start log; never touch the real one.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.start_log = Path(directory.name) / 'starts.jsonl'
        log_patch = patch('inference.executor.executor.DEFAULT_START_LOG', self.start_log)
        log_patch.start()
        self.addCleanup(log_patch.stop)

    def test_camera_applies_collection_config_defaults(self):
        from data_collection.collection import config

        args = parse_args(['--predictor', 'localhost'])
        for option, setting in (
            ('cam_width', 'TASK_CAMERA_WIDTH'), ('cam_height', 'TASK_CAMERA_HEIGHT'),
            ('cam_fps', 'TASK_CAMERA_FPS'), ('cam_flip', 'TASK_CAMERA_FLIP'),
            ('cam_exposure_us', 'TASK_CAMERA_EXPOSURE_US'),
            ('cam_iso', 'TASK_CAMERA_ISO'),
            ('cam_white_balance_k', 'TASK_CAMERA_WHITE_BALANCE_K'),
        ):
            self.assertEqual(getattr(args, option), getattr(config, setting))

        dai = Mock()
        pipeline = dai.Pipeline.return_value
        camera_node = pipeline.create.return_value.build.return_value
        with patch('data_collection.collection.depthai_utils.import_depthai', return_value=dai), \
             patch('data_collection.collection.depthai_utils.require_depthai_device'), \
             patch('data_collection.collection.depthai_utils.apply_camera_control') as control, \
             patch.object(OakCamera, 'read'):
            camera = OakCamera(args)
            pipeline.create.return_value.build.assert_called_once_with(sensorFps=config.TASK_CAMERA_FPS)
            control.assert_called_once_with(
                camera_node, exposure_us=config.TASK_CAMERA_EXPOSURE_US,
                iso=config.TASK_CAMERA_ISO, white_balance_k=config.TASK_CAMERA_WHITE_BALANCE_K,
            )
            camera_node.requestOutput.assert_called_once_with(
                (config.TASK_CAMERA_WIDTH, config.TASK_CAMERA_HEIGHT), dai.ImgFrame.Type.NV12,
                fps=config.TASK_CAMERA_FPS,
            )
            self.assertEqual(camera.flip, config.TASK_CAMERA_FLIP)
            camera.close()
            pipeline.stop.assert_called_once()

    def test_show_previews_newest_frame_and_q_stops(self):
        camera = OakCamera.__new__(OakCamera)
        camera.cv2, camera.flip, camera.timeout = Mock(), 'none', 1.0
        camera.window = 'wrist'
        camera.flip_frame = lambda cv2, frame, flip: frame
        old, new = Mock(), Mock()
        new.getCvFrame.return_value = 'newest'
        camera.queue = Mock()
        camera.queue.tryGet.side_effect = [old, new, None, None]
        camera.cv2.waitKey.return_value = -1
        camera.preview()
        camera.cv2.imshow.assert_called_once_with('wrist', 'newest')
        camera.preview()  # Empty queue: keep the last frame on screen.
        camera.cv2.imshow.assert_called_once()
        camera.cv2.waitKey.return_value = ord('q')
        camera.queue.tryGet.side_effect = [new, None]
        with self.assertRaises(KeyboardInterrupt):
            camera.preview()
        camera.window = None
        camera.queue.tryGet.side_effect = AssertionError('preview must not consume frames when hidden')
        camera.preview()

    def test_start_move_ramps_to_named_start_before_policy_speed(self):
        for extra, name in (([], config.DEFAULT_EXECUTOR_START), (['--start-home', 'quest_0'], 'quest_0')):
            args = parse_args(['--predictor', 'localhost', '--piper-speed', '40',
                               '--start-jitter-mm', '0', '--start-jitter-deg', '0', *extra])
            robot = RobotIO.__new__(RobotIO)
            robot.piper = Mock()
            robot.piper.get_eef_pose.return_value = [0.4, -0.2, 0.2, 3.0, 0.5, 1.5]
            robot.hand = Mock()
            robot.hand.clip_positions.side_effect = lambda q: q
            events = []
            go_home_events = lambda *a: events.append('arm')
            robot.hand.set_positions_blocking.side_effect = lambda *a, **k: events.append('hand')
            with patch('builtins.input', return_value='') as prompt, \
                 patch('scripts.piper_api.replay_trajectory.read_joint_angles', return_value=[0.0] * 6), \
                 patch('scripts.teleop.quest_teleop.go_home', side_effect=go_home_events) as go_home, \
                 patch('inference.executor.executor.time.sleep'), \
                 contextlib.redirect_stdout(io.StringIO()):
                robot.move_to_start(args)
            prompt.assert_called_once()
            self.assertEqual(events, ['arm', 'hand'])
            np.testing.assert_allclose(robot.hand.set_positions_blocking.call_args.args[0],
                                       config.HAND_START_POSITIONS_RAD[name])
            robot.piper.set_position_mode.assert_called_once()
            robot.piper.set_speed_percent.assert_called_once_with(100)
            np.testing.assert_allclose(go_home.call_args.args[1], np.radians(config.PIPER_START_JOINTS_DEG[name]))
            self.assertEqual(go_home.call_args.args[2], 10.0)
        self.assertEqual(config.DEFAULT_EXECUTOR_START, 'home_4c')
        self.assertEqual(set(config.HAND_START_POSITIONS_RAD), set(config.PIPER_START_JOINTS_DEG))

    def test_hand_start_timeout_warns_and_no_hand_start_skips(self):
        robot = RobotIO.__new__(RobotIO)
        robot.hand = Mock()
        robot.hand.clip_positions.side_effect = lambda q: q
        robot.hand.set_positions_blocking.side_effect = TimeoutError
        robot.hand.read_pos.return_value = np.zeros(13)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            robot.move_hand_to_start('home_4b')
        self.assertIn('WARNING: hand not within', output.getvalue())
        self.assertIn('motor 2 0.628 rad', output.getvalue())
        args = parse_args(['--predictor', 'localhost', '--no-hand-start'])
        robot.piper = Mock()
        robot.piper.get_eef_pose.return_value = None
        robot.hand.reset_mock()
        with patch('builtins.input', return_value=''), \
             patch('scripts.piper_api.replay_trajectory.read_joint_angles', return_value=[0.0] * 6), \
             patch('scripts.teleop.quest_teleop.go_home'), \
             patch('inference.executor.executor.time.sleep'), \
             contextlib.redirect_stdout(io.StringIO()):
            robot.move_to_start(args)
        robot.hand.set_positions_blocking.assert_not_called()

    def test_start_jitter_stays_within_radius_angle_and_joint_checks(self):
        from data_collection.replay_actions import Limits, check_joints, piper_model
        fk, bounds = piper_model()
        home = config.PIPER_START_JOINTS_DEG['home_4b']
        start = fk(np.radians(home))
        rng = np.random.default_rng(0)
        offsets = []
        for _ in range(10):
            joints, offset_m, angle_deg = jitter_start_joints(home, 0.03, 10.0, rng)
            pose = fk(np.radians(joints))
            moved = np.linalg.norm(pose[:3] - start[:3])
            turned = np.degrees((Rotation.from_euler('xyz', pose[3:])
                                 * Rotation.from_euler('xyz', start[3:]).inv()).magnitude())
            self.assertLessEqual(moved, 0.03 + 5e-4)
            self.assertLessEqual(turned, 10.2)
            self.assertAlmostEqual(moved, offset_m, delta=5e-4)
            self.assertAlmostEqual(turned, angle_deg, delta=0.2)
            check_joints(np.radians(joints), bounds, fk, Limits())
            offsets.append(offset_m)
        self.assertGreater(len(set(np.round(offsets, 4))), 5)
        joints, offset_m, angle_deg = jitter_start_joints(home, 0, 0, rng)
        np.testing.assert_allclose(joints, home)
        self.assertEqual((offset_m, angle_deg), (0.0, 0.0))
        args = parse_args(['--predictor', 'localhost'])
        self.assertEqual((args.start_jitter_mm, args.start_jitter_deg), (0.0, 0.0))

    def test_every_start_is_logged_and_start_id_repeats_it(self):
        def start(argv):
            args = parse_args(['--predictor', 'localhost', *argv])
            robot = RobotIO.__new__(RobotIO)
            robot.piper = Mock()
            robot.piper.get_eef_pose.return_value = [0.434, -0.224, 0.222, 2.8, 0.54, 1.58]
            robot.hand = Mock()
            robot.hand.clip_positions.side_effect = lambda q: q
            output = io.StringIO()
            with patch('builtins.input', return_value=''), \
                 patch('scripts.piper_api.replay_trajectory.read_joint_angles', return_value=[0.0] * 6), \
                 patch('scripts.teleop.quest_teleop.go_home') as go_home, \
                 patch('inference.executor.executor.time.sleep'), \
                 contextlib.redirect_stdout(output):
                robot.move_to_start(args)
            return go_home.call_args.args[1], robot.hand.set_positions_blocking.call_args.args[0], output.getvalue()

        first, _, text = start(['--start-home', 'home_4a', '--start-jitter-mm', '20', '--start-jitter-deg', '5'])
        self.assertIn('Start #0 logged', text)
        record = json.loads(self.start_log.read_text().splitlines()[0])
        self.assertEqual(record['base'], 'home_4a')
        self.assertEqual(record['hand_start'], 'home_4a')
        self.assertGreater(record['random_offset_mm'], 0)
        np.testing.assert_allclose(record['joints_deg'], np.degrees(first), atol=1e-3)
        self.assertEqual(record['flange_xyz_mm'], [434.0, -224.0, 222.0])
        # Repeating #0 reproduces its joints and hand start exactly, even with --start-home left at its default.
        repeated, hand, text = start(['--start-id', '0'])
        np.testing.assert_allclose(repeated, np.radians(record['joints_deg']))
        np.testing.assert_allclose(hand, config.HAND_START_POSITIONS_RAD['home_4a'])
        self.assertIn('start #0', text)
        self.assertIn('Start #1 logged', text)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--predictor', 'localhost', '--start-id', '5'])

    def test_start_move_rejects_targets_outside_controller_limits(self):
        args = parse_args(['--predictor', 'localhost', '--start-joints-deg', '0', '0', '10', '0', '30', '0'])
        robot = RobotIO.__new__(RobotIO)
        robot.piper = Mock()
        with self.assertRaises(ValueError):
            robot.move_to_start(args)
        robot.piper.move_joint.assert_not_called()

    def test_hardware_setup_uses_trimorph_replay_configuration(self):
        args = parse_args(['--predictor', 'localhost'])
        hand = Mock()
        hand.motor_ids = list(range(13))
        hand.read_pos.return_value = np.zeros(13)
        piper = Mock()
        piper.connect.return_value = True
        piper.enable.return_value = True
        events = []
        hand.set_positions.side_effect = lambda *a, **k: events.append('hold')
        hand.enable_torque.side_effect = lambda: events.append('enable')
        hand_class = Mock(return_value=hand)
        with patch('inference.executor.executor.load_hand_class', return_value=hand_class), \
             patch('data_collection.collection.hand_common.load_hand_config', return_value=SimpleConfig()), \
             patch('data_collection.collection.hand_common.verify_connected_hand'), \
             patch('scripts.piper_api.piper_api.PiperAPI', return_value=piper) as factory, \
             patch.object(RobotIO, 'move_to_start', side_effect=lambda _: events.append('start')):
            robot = RobotIO(args)
            self.assertEqual(hand_class.call_args.kwargs['config'].operating_mode, 3)
            self.assertEqual(factory.call_args.kwargs['channel'], args.piper_channel)
            hand.configure.assert_called_once_with(enable_torque=False)
            self.assertEqual(events, ['hold', 'enable', 'start'])
            piper.set_speed_percent.assert_called_once_with(args.piper_speed)
            robot.close()
        hand.shutdown.assert_called_once()
        piper.disconnect.assert_called_once()

    def test_pose_roundtrip_preserves_rotation(self):
        for angles in ([0, 0, 0], [.4, -.8, 1.2], [2.9, 1.2, -2.8]):
            original = np.array([.2, -.1, .3, *angles])
            policy = pose_to_policy(original)
            returned = pose_to_piper(policy)
            np.testing.assert_allclose(returned[:3], original[:3])
            np.testing.assert_allclose(
                Rotation.from_euler('xyz', returned[3:]).as_matrix(),
                Rotation.from_euler('xyz', original[3:]).as_matrix(), atol=1e-12,
            )

    def test_read_state_uses_13_calibrated_motor_positions(self):
        robot = RobotIO.__new__(RobotIO)
        pose = [.1, .2, .3, .4, .5, .6]
        positions = np.arange(13) / 10
        robot.hand = Mock()
        robot.piper = Mock()
        robot.hand.read_pos.return_value = positions
        robot.piper.get_eef_pose.return_value = pose
        robot.state_anchor, robot.commanded_pose = 'command', None  # nothing commanded yet
        state = robot.read_state()
        np.testing.assert_allclose(state[:6], pose_to_policy(pose), atol=1e-7)
        np.testing.assert_allclose(state[6:], positions)
        robot.hand.read_joint_pos.assert_not_called()

    def test_state_anchor_uses_commanded_pose_unless_arm_is_far_from_it(self):
        robot = RobotIO.__new__(RobotIO)
        robot.hand = Mock()
        robot.hand.read_pos.return_value = np.zeros(13)
        robot.piper = Mock()
        robot.state_anchor, robot.anchor_max_mm, robot.anchor_max_deg = 'command', 50.0, 20.0
        commanded = np.array([.4, -.2, .2, 3.0, .5, 1.5])
        robot.commanded_pose = commanded
        lagging = commanded + [.02, 0, 0, 0, .05, 0]  # 20 mm and ~3 deg behind
        robot.piper.get_eef_pose.return_value = lagging.tolist()
        np.testing.assert_allclose(robot.read_state()[:6], pose_to_policy(commanded), atol=1e-6)
        self.assertTrue(robot.state_note.startswith('command'))
        far = commanded + [.08, 0, 0, 0, 0, 0]  # 80 mm away: fall back to the measured pose
        robot.piper.get_eef_pose.return_value = far.tolist()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            state = robot.read_state()
        np.testing.assert_allclose(state[:6], pose_to_policy(far), atol=1e-6)
        self.assertIn('FALLBACK', robot.state_note)
        self.assertIn('WARNING', output.getvalue())
        robot.state_anchor = 'measured'
        robot.piper.get_eef_pose.return_value = lagging.tolist()
        np.testing.assert_allclose(robot.read_state()[:6], pose_to_policy(lagging), atol=1e-6)

    def test_joint_control_anchor_is_fk_of_sent_joints_and_resets_at_start(self):
        from inference.executor.joint_control import JointController
        robot = RobotIO.__new__(RobotIO)
        robot.hand, robot.piper = Mock(), Mock()
        robot.dt, robot.joint_control, robot.substeps = 1 / 15, JointController(), 1
        start = np.radians(config.PIPER_START_JOINTS_DEG['home_cube1'])
        robot.piper.get_joint_angles.return_value = start.tolist()
        target = robot.joint_control.fk(start + np.radians([0, 2, -3, 0, 0, 0]))
        info = robot.apply_action(np.r_[pose_to_policy(target), np.zeros(13)])
        np.testing.assert_allclose(robot.commanded_pose, robot.joint_control.fk(info['joints']))
        with patch.object(RobotIO, 'move_to_start'):
            robot.return_to_start(SimpleNamespace(piper_speed=100))
        self.assertIsNone(robot.commanded_pose)
        self.assertIsNone(robot.joint_control.q)

    def test_command_converts_pose_without_retargeting_hand(self):
        robot = RobotIO.__new__(RobotIO)
        robot.hand = Mock()
        robot.piper = Mock()
        robot.joint_control = None  # --arm-control pose
        action = np.r_[pose_to_policy([.1, .2, .3, .4, -.5, .6]), np.arange(13) / 10]
        self.assertIsNone(robot.apply_action(action))
        np.testing.assert_allclose(robot.piper.move_eef.call_args.args[0], [.1, .2, .3, .4, -.5, .6])
        np.testing.assert_array_equal(robot.hand.set_positions.call_args.args[0], action[6:])
        self.assertEqual(robot.hand.set_positions.call_args.kwargs, {'clip': True})
        robot.hand.set_positions_blocking.assert_not_called()

    def test_joint_control_sends_move_j_from_measured_joints(self):
        from inference.executor.joint_control import JointController
        robot = RobotIO.__new__(RobotIO)
        robot.hand = Mock()
        robot.piper = Mock()
        robot.dt, robot.substeps = 1 / 15, 1
        robot.joint_control = JointController()
        start = np.radians(config.PIPER_START_JOINTS_DEG['home_cube1'])
        robot.piper.get_joint_angles.return_value = start.tolist()
        pose = robot.joint_control.fk(start)
        action = np.r_[pose_to_policy(pose), np.zeros(13)]
        info = robot.apply_action(action)
        robot.piper.move_eef.assert_not_called()
        self.assertEqual(robot.piper.move_joint.call_args.kwargs, {'mode': 'smooth'})
        np.testing.assert_allclose(robot.piper.move_joint.call_args.args[0], start, atol=1e-6)
        self.assertLess(info['pos_err_mm'], 0.01)
        # A pedal reset restarts from the measured joints.
        robot.joint_control.q = None
        robot.piper.get_joint_angles.reset_mock()
        robot.apply_action(action)
        robot.piper.get_joint_angles.assert_called_once()

    def test_joint_targets_are_interpolated_over_the_action_period(self):
        from inference.executor.joint_control import JointController
        robot = RobotIO.__new__(RobotIO)
        robot.hand, robot.piper = Mock(), Mock()
        robot.dt, robot.joint_control, robot.substeps = 1 / 15, JointController(), 4
        start = np.radians(config.PIPER_START_JOINTS_DEG['home_cube1'])
        robot.joint_control.reset(start)
        target = robot.joint_control.fk(start + np.radians([1, 2, -3, 1, -1, 2]))
        info = robot.apply_action(np.r_[pose_to_policy(target), np.zeros(13)])
        for k in (2, 3, 4):
            robot.send_substep(k / 4)
        sent = np.array([c.args[0] for c in robot.piper.move_joint.call_args_list])
        expected = start + np.arange(1, 5)[:, None] / 4 * (info['joints'] - start)
        np.testing.assert_allclose(sent, expected, atol=1e-9)
        np.testing.assert_allclose(sent[-1], info['joints'])

    def test_run_loop_sends_substeps_between_actions(self):
        robot = FakeRobot()
        robot.substeps = 4
        calls = []
        robot.apply_action = Mock(side_effect=lambda action: calls.append('action'))
        robot.send_substep = Mock(side_effect=lambda fraction: calls.append(fraction))
        chunk = np.zeros((2, 19), dtype=np.float32)
        with patch.object(protocol, 'send_obs'), \
             patch.object(protocol, 'recv_action', return_value=(0, chunk, {})), \
             contextlib.redirect_stdout(io.StringIO()):
            run(SimpleNamespace(hz=1000, max_cycles=1, jpeg_quality=85), robot, FakeCamera(), Mock())
        self.assertEqual(calls, ['action', .5, .75, 1.0, 'action', .5, .75, 1.0])

    def test_invalid_action_never_reaches_hardware(self):
        robot = RobotIO.__new__(RobotIO)
        robot.hand = Mock()
        robot.piper = Mock()
        for action in (np.zeros(16), np.full(19, np.nan)):
            with self.assertRaises(ValueError):
                robot.apply_action(action)
        robot.hand.set_positions.assert_not_called()
        robot.piper.move_eef.assert_not_called()

    def test_sync_loop_limits_playback_to_selected_prefix(self):
        chunk = np.zeros((5, 19))
        chunk[:, 6] = np.arange(5) / 10
        robot = FakeRobot()
        robot.apply_action = Mock(wraps=robot.apply_action)
        view = Mock()
        args = SimpleNamespace(hz=1000, max_cycles=1, jpeg_quality=85, execute_steps=2)
        with patch.object(protocol, 'send_obs'), \
             patch.object(protocol, 'recv_action', return_value=(0, chunk, {})), \
             contextlib.redirect_stdout(io.StringIO()):
            run(args, robot, FakeCamera(), Mock(), view=view)
        self.assertEqual(robot.apply_action.call_count, 2)
        self.assertAlmostEqual(robot.state[6], .1)
        np.testing.assert_array_equal(view.set_chunk.call_args.args[0], chunk[:2, :3])

    def test_socket_loop_executes_multiple_chunks(self):
        client, server = socket.socketpair()
        client.settimeout(2)
        server.settimeout(2)
        robot = FakeRobot()
        robot.apply_action = Mock(wraps=robot.apply_action)
        errors = []
        target = np.arange(19, dtype=np.float32) / 100
        scaled_target = target.astype(np.float64)
        scaled_target[6:] *= 1.1
        action_alpha = np.r_[np.full(6, 0.2), np.full(13, 0.8)]
        def predict():
            try:
                for seq in range(2):
                    received_seq, image, state, _ = protocol.recv_obs(server)
                    self.assertEqual(received_seq, seq)
                    self.assertEqual(image.shape, (240, 320, 3))
                    expected_state = scaled_target * (1 - (1 - action_alpha) ** (2 * seq))
                    np.testing.assert_allclose(state, expected_state, atol=1e-7)
                    protocol.send_action(server, seq, np.tile(target, (2, 1)), infer_ms=1.5)
            except BaseException as exc:
                errors.append(exc)
            finally:
                server.close()
        thread = threading.Thread(target=predict)
        thread.start()
        try:
            args = SimpleNamespace(hz=1000, max_cycles=2, jpeg_quality=85)
            with patch('inference.executor.executor.ARM_ACTION_EMA_ALPHA', 0.2), \
                 patch('inference.executor.executor.HAND_ACTION_EMA_ALPHA', 0.8), \
                 patch('inference.executor.executor.HAND_ACTION_SCALE', 1.1), \
                 contextlib.redirect_stdout(io.StringIO()):
                run(args, robot, FakeCamera(), client)
        finally:
            client.close()
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        if errors:
            raise errors[0]
        # Start from feedback, smooth every coordinate, and retain EMA history
        # across the boundary between the two chunks.
        self.assertEqual(robot.apply_action.call_count, 4)
        for step, call in enumerate(robot.apply_action.call_args_list, start=1):
            expected = scaled_target * (1 - (1 - action_alpha) ** step)
            np.testing.assert_allclose(call.args[0], expected, atol=1e-7)
        np.testing.assert_allclose(robot.read_state(), expected, atol=1e-7)

    def test_pedal_interrupts_chunk_returns_to_start_and_resumes(self):
        # Polls: setup drain, cycle 0 start, action 0, action 1 (press), drain after
        # return, wait (idle), wait (press = resume), cycle 1 start, then cycle 1 actions.
        pedal = Mock()
        presses = iter([[1], [], [], [1], [1], [], [1]])
        pedal.poll.side_effect = lambda: next(presses, [])
        chunk = np.zeros((3, 19))
        robot = FakeRobot()
        robot.apply_action = Mock()
        robot.return_to_start = Mock()
        args = SimpleNamespace(hz=1000, max_cycles=2, jpeg_quality=85)
        output = io.StringIO()
        with patch('inference.executor.executor.protocol.send_obs'), \
             patch('inference.executor.executor.protocol.recv_action',
                   side_effect=[(0, chunk, {}), (1, chunk, {})]), \
             patch('inference.executor.executor.time.sleep'), \
             contextlib.redirect_stdout(output):
            run(args, robot, FakeCamera(), Mock(), pedal)
        robot.return_to_start.assert_called_once_with(args)
        # Cycle 0 stops after one action; cycle 1 runs its whole chunk.
        self.assertEqual(robot.apply_action.call_count, 1 + len(chunk))
        self.assertIn('interrupted after 1 actions', output.getvalue())
        self.assertIn('resuming the policy', output.getvalue())

    def test_trajectory_view_gets_chunk_and_pose_after_each_action(self):
        chunk = np.zeros((3, 19))
        chunk[:, 0] = [0.01, 0.02, 0.03]
        view = Mock()
        args = SimpleNamespace(hz=1000, max_cycles=1, jpeg_quality=85)
        with patch('inference.executor.executor.protocol.send_obs'), \
             patch('inference.executor.executor.protocol.recv_action', return_value=(0, chunk, {})), \
             patch('inference.executor.executor.ARM_ACTION_EMA_ALPHA', 1.0), \
             contextlib.redirect_stdout(io.StringIO()):
            run(args, FakeRobot(), FakeCamera(), Mock(), view=view)
        np.testing.assert_allclose(view.set_chunk.call_args.args[0], chunk[:, :3])
        self.assertEqual(view.update.call_count, len(chunk))
        self.assertAlmostEqual(view.update.call_args.args[0][0], 0.03)

    def test_large_jump_rejects_whole_chunk_before_any_action(self):
        for positions, source in (
            ([[0.16, 0, 0]], 'observation'),
            ([[0.01, 0, 0], [0.17, 0, 0]], 'action 0'),
            ([[0, 0, 0], [0.1, 0.1, 0.1]], 'action 0'),
        ):
            with self.subTest(positions=positions):
                chunk = np.zeros((len(positions), 19))
                chunk[:, :3] = positions
                robot = FakeRobot()
                robot.apply_action = Mock()
                output = io.StringIO()
                args = SimpleNamespace(hz=1000, max_cycles=2, jpeg_quality=85)
                with patch('inference.executor.executor.protocol.send_obs'), \
                     patch('inference.executor.executor.protocol.recv_action', return_value=(0, chunk, {})) as receive, \
                     contextlib.redirect_stdout(output):
                    run(args, robot, FakeCamera(), Mock())
                robot.apply_action.assert_not_called()
                receive.assert_called_once()
                self.assertIn('WARNING: refusing entire chunk', output.getvalue())
                self.assertIn(f'from {source}', output.getvalue())
                self.assertIn('15 cm', output.getvalue())

    def test_jump_at_limit_and_gradual_motion_are_allowed(self):
        for positions in (
            [[MAX_EEF_JUMP_M, 0, 0]],
            [[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]],
        ):
            with self.subTest(positions=positions):
                chunk = np.zeros((len(positions), 19))
                chunk[:, :3] = positions
                robot = FakeRobot()
                robot.apply_action = Mock()
                args = SimpleNamespace(hz=10000, max_cycles=1, jpeg_quality=85)
                with patch('inference.executor.executor.protocol.send_obs'), \
                     patch('inference.executor.executor.protocol.recv_action', return_value=(0, chunk, {})), \
                     contextlib.redirect_stdout(io.StringIO()):
                    run(args, robot, FakeCamera(), Mock())
                self.assertEqual(robot.apply_action.call_count, len(chunk))

    def test_wrong_sequence_does_not_execute(self):
        client, server = socket.socketpair()
        robot = FakeRobot()
        try:
            protocol.send_action(server, 99, np.ones((1, 19)), infer_ms=1)
            with self.assertRaisesRegex(ValueError, 'sequence'):
                run(SimpleNamespace(hz=15, max_cycles=1, jpeg_quality=85), robot, FakeCamera(), client)
            np.testing.assert_array_equal(robot.read_state(), np.zeros(19))
        finally:
            client.close()
            server.close()

    def test_bad_wire_outputs_rejected(self):
        for chunk in (np.zeros((2, 16)), np.zeros((0, 19)), np.full((1, 19), np.inf)):
            with self.assertRaises(ValueError):
                protocol.send_action(Mock(), 0, chunk, infer_ms=1)

    def test_cli_rejects_invalid_timing(self):
        for option, value in (('--hz', '0'), ('--timeout', 'nan'), ('--max-cycles', '-1'),
                              ('--start-speed-deg-s', '45')):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--predictor', 'localhost', option, value])


if __name__ == '__main__':
    unittest.main()
