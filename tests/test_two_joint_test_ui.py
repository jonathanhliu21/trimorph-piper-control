import queue
import unittest

import numpy as np

from data_collection.collection.contact_detection import (
    ContactState,
    ContactStateMachine,
)
from scripts.trimorph_api_test.two_joint_test_ui import (
    HandWorker,
    STREAM_CLOSE_ANGLE_RAD,
    STREAM_OPEN_ANGLE_RAD,
    _stream_sample,
    _ui_order_indices,
    _validated_default_targets,
)


class HandWorkerContactTests(unittest.TestCase):
    class FakeHand:
        def __init__(self):
            self.positions = np.asarray([0.0, 0.0])
            self.velocities = np.asarray([0.0, 0.0])
            self.currents = np.asarray([12.0, 13.0])
            self.last_targets = None

        def clip_positions(self, positions):
            return np.clip(positions, -1.5, 0.05)

        def set_positions(self, positions, clip=True):
            self.last_targets = np.asarray(positions)

        def read_pos_vel_cur(self):
            return self.positions, self.velocities, self.currents

    def test_shared_state_machine_suppresses_contact_during_motion(self):
        events = queue.Queue()
        worker = HandWorker(queue.Queue(), events)
        worker.hand = self.FakeHand()

        worker._handle_command("targets", (-0.10, 0.0))
        worker._control_tick()
        telemetry = events.get_nowait().payload

        self.assertEqual(telemetry.contacts, (False, False))
        self.assertNotEqual(telemetry.candidate_feedback[0], 0.0)
        self.assertEqual(telemetry.contact_feedback[0], 0.0)
        self.assertIs(worker.contact_detectors[0].state, ContactState.MOVING)

    def test_stationary_tracking_error_becomes_contact(self):
        events = queue.Queue()
        worker = HandWorker(queue.Queue(), events)
        worker.hand = self.FakeHand()
        worker.targets = np.asarray([-0.10, 0.0])
        worker.contact_detectors = tuple(
            ContactStateMachine(
                settle_time_s=0.0,
                persistence_time_s=0.0,
                min_engaged_time_s=0.0,
                exit_persistence_time_s=0.0,
            )
            for _ in range(2)
        )

        worker._control_tick()
        first = events.get_nowait().payload
        worker._control_tick()
        second = events.get_nowait().payload
        worker._control_tick()
        third = events.get_nowait().payload

        self.assertEqual(first.contacts, (False, False))
        self.assertEqual(second.contacts, (False, False))
        self.assertEqual(third.contacts, (True, False))

    def test_commands_and_telemetry_are_mapped_by_motor_id(self):
        events = queue.Queue()
        worker = HandWorker(queue.Queue(), events)
        hand = self.FakeHand()
        # HandConfig.subset preserves YAML order (PIP 4, MCP 5), while the UI
        # presents MCP first and PIP second.
        worker.ui_to_hand_indices = (1, 0)
        worker.hand = hand
        worker.targets = np.asarray([-0.10, -0.20])  # MCP, PIP
        hand.positions = np.asarray([-0.25, -0.15])  # PIP, MCP

        worker._control_tick()
        telemetry = events.get_nowait().payload

        np.testing.assert_allclose(hand.last_targets, (-0.20, -0.10))
        self.assertEqual(telemetry.positions, (-0.15, -0.25))


class StreamSequenceTests(unittest.TestCase):
    def test_stream_moves_open_to_close_and_back_to_open(self):
        duration_s = 2.0

        start, _, start_complete = _stream_sample(0.0, duration_s)
        closed, _, closed_complete = _stream_sample(duration_s, duration_s)
        opened, _, opened_complete = _stream_sample(2.0 * duration_s, duration_s)
        finished, speed, finished_complete = _stream_sample(2.1 * duration_s, duration_s)

        np.testing.assert_allclose(start, STREAM_OPEN_ANGLE_RAD)
        np.testing.assert_allclose(closed, STREAM_CLOSE_ANGLE_RAD)
        np.testing.assert_allclose(opened, STREAM_OPEN_ANGLE_RAD)
        np.testing.assert_allclose(finished, STREAM_OPEN_ANGLE_RAD)
        self.assertFalse(start_complete)
        self.assertFalse(closed_complete)
        self.assertFalse(opened_complete)
        self.assertTrue(finished_complete)
        self.assertEqual(speed, 0.0)

    def test_stream_rejects_non_positive_duration(self):
        for duration_s in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(duration_s=duration_s):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    _stream_sample(0.0, duration_s)


class JointLimitTests(unittest.TestCase):
    def test_ui_order_maps_mcp_5_before_pip_4(self):
        self.assertEqual(_ui_order_indices([4, 5], 5, 4), (1, 0))

    def test_zero_defaults_are_validated_against_configured_limits(self):
        targets = _validated_default_targets((-1.5, -1.7), (0.05, 0.05))

        np.testing.assert_array_equal(targets, (0.0, 0.0))

    def test_zero_default_outside_joint_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "PIP"):
            _validated_default_targets((-1.5, 0.1), (0.05, 1.0))


if __name__ == "__main__":
    unittest.main()
