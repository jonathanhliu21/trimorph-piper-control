"""Hardware-free coverage of the deployment coordinate and wire contracts."""
import contextlib
from dataclasses import dataclass
import io
import socket
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from inference import protocol
from inference.executor.executor import (
    FakeCamera, FakeRobot, RobotIO, parse_args, pose_to_piper, pose_to_policy, run,
)


@dataclass
class SimpleConfig:
    operating_mode: int = 5


class InferenceTests(unittest.TestCase):
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
             patch('scripts.piper_api.piper_api.PiperAPI', return_value=piper) as factory:
            robot = RobotIO(args)
            self.assertEqual(hand_class.call_args.kwargs['config'].operating_mode, 3)
            self.assertEqual(factory.call_args.kwargs['channel'], args.piper_channel)
            hand.configure.assert_called_once_with(enable_torque=False)
            self.assertEqual(events, ['hold', 'enable'])
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
        state = robot.read_state()
        np.testing.assert_allclose(state[:6], pose_to_policy(pose), atol=1e-7)
        np.testing.assert_allclose(state[6:], positions)
        robot.hand.read_joint_pos.assert_not_called()

    def test_command_converts_pose_without_retargeting_hand(self):
        robot = RobotIO.__new__(RobotIO)
        robot.hand = Mock()
        robot.piper = Mock()
        action = np.r_[pose_to_policy([.1, .2, .3, .4, -.5, .6]), np.arange(13) / 10]
        robot.apply_action(action)
        np.testing.assert_allclose(robot.piper.move_eef.call_args.args[0], [.1, .2, .3, .4, -.5, .6])
        np.testing.assert_array_equal(robot.hand.set_positions.call_args.args[0], action[6:])
        self.assertEqual(robot.hand.set_positions.call_args.kwargs, {'clip': True})
        robot.hand.set_positions_blocking.assert_not_called()

    def test_invalid_action_never_reaches_hardware(self):
        robot = RobotIO.__new__(RobotIO)
        robot.hand = Mock()
        robot.piper = Mock()
        for action in (np.zeros(16), np.full(19, np.nan)):
            with self.assertRaises(ValueError):
                robot.apply_action(action)
        robot.hand.set_positions.assert_not_called()
        robot.piper.move_eef.assert_not_called()

    def test_socket_loop_executes_multiple_chunks(self):
        client, server = socket.socketpair()
        client.settimeout(2)
        server.settimeout(2)
        robot = FakeRobot()
        errors = []
        target = np.arange(19, dtype=np.float32) / 100
        def predict():
            try:
                for seq in range(2):
                    received_seq, image, state, _ = protocol.recv_obs(server)
                    self.assertEqual(received_seq, seq)
                    self.assertEqual(image.shape, (240, 320, 3))
                    np.testing.assert_allclose(state, np.zeros(19) if seq == 0 else target)
                    protocol.send_action(server, seq, np.tile(target, (2, 1)), infer_ms=1.5)
            except BaseException as exc:
                errors.append(exc)
            finally:
                server.close()
        thread = threading.Thread(target=predict)
        thread.start()
        try:
            args = SimpleNamespace(hz=1000, max_cycles=2, jpeg_quality=85)
            with contextlib.redirect_stdout(io.StringIO()):
                run(args, robot, FakeCamera(), client)
        finally:
            client.close()
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        if errors:
            raise errors[0]
        np.testing.assert_allclose(robot.read_state(), target)

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
        for option, value in (('--hz', '0'), ('--timeout', 'nan'), ('--max-cycles', '-1')):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--predictor', 'localhost', option, value])


if __name__ == '__main__':
    unittest.main()
