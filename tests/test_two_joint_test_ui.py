import queue
import unittest

import numpy as np

from data_collection.collection.contact_detection import (
    ContactState,
    ContactStateMachine,
)
from scripts.trimorph_api_test.hand_ui import (
    AB_AD_INDICES,
    ContactDetectorSettings,
    FLEXION_INDICES,
    HandWorker,
    JOINT_COUNT,
    STREAM_CLOSE_ANGLE_RAD,
    STREAM_OPEN_ANGLE_RAD,
    UI_MOTOR_IDS,
    _build_contact_detectors,
    _contact_circle_radius,
    _stream_sample,
    _threshold_normalized_joint_difference,
    _ui_order_indices,
    _validated_default_targets,
)


class ContactTuningTests(unittest.TestCase):
    def test_custom_settings_are_applied_to_all_detectors(self):
        settings = ContactDetectorSettings(
            max_velocity_rad_s=2.0,
            motion_current_threshold_ma=275.0,
            settle_time_s=0.2,
            persistence_time_s=0.3,
            entry_feedback_threshold=0.4,
            min_engaged_time_s=0.5,
            exit_persistence_time_s=0.6,
        )

        detectors = _build_contact_detectors(settings)

        self.assertEqual(len(detectors), JOINT_COUNT)
        self.assertTrue(all(detector.max_velocity_rad_s == 2.0 for detector in detectors))
        self.assertTrue(
            all(detector.motion_current_threshold_ma == 275.0 for detector in detectors)
        )
        self.assertTrue(all(detector.settle_time_s == 0.2 for detector in detectors))
        self.assertTrue(all(detector.persistence_time_s == 0.3 for detector in detectors))
        self.assertTrue(
            all(detector.entry_feedback_threshold == 0.4 for detector in detectors)
        )
        self.assertTrue(all(detector.min_engaged_time_s == 0.5 for detector in detectors))
        self.assertTrue(
            all(detector.exit_persistence_time_s == 0.6 for detector in detectors)
        )

    def test_negative_or_non_finite_settings_are_rejected(self):
        for value in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite, non-negative"):
                    _build_contact_detectors(
                        ContactDetectorSettings(max_velocity_rad_s=value)
                    )

    def test_contact_circle_grows_with_feedback_and_is_bounded(self):
        self.assertLess(_contact_circle_radius(0.02), _contact_circle_radius(0.10))
        self.assertEqual(_contact_circle_radius(100.0), 120.0)

    def test_joint_difference_is_normalized_by_effective_entry_threshold(self):
        normalized = _threshold_normalized_joint_difference(
            np.asarray([0.05, -0.10, 0.025]), ContactDetectorSettings()
        )

        np.testing.assert_allclose(normalized, (1.0, 2.0, 0.5))


class HandWorkerContactTests(unittest.TestCase):
    class FakeHand:
        def __init__(self):
            self.positions = np.zeros(JOINT_COUNT)
            self.velocities = np.zeros(JOINT_COUNT)
            self.currents = np.arange(12.0, 12.0 + JOINT_COUNT)
            self.last_targets = None

        def clip_positions(self, positions):
            return np.clip(positions, -1.5, 0.05)

        def set_positions(self, positions, clip=True):
            self.last_targets = np.asarray(positions)

        def read_pos_vel_cur(self):
            return self.positions, self.velocities, self.currents

    def test_motion_below_current_threshold_does_not_detect_contact(self):
        events = queue.Queue()
        worker = HandWorker(queue.Queue(), events)
        worker.hand = self.FakeHand()

        targets = np.zeros(JOINT_COUNT)
        targets[0] = -0.10
        worker._handle_command("targets", targets)
        worker._control_tick()
        telemetry = events.get_nowait().payload

        self.assertEqual(telemetry.contacts, (False,) * JOINT_COUNT)
        self.assertNotEqual(telemetry.candidate_feedback[0], 0.0)
        self.assertEqual(telemetry.contact_feedback[0], 0.0)
        self.assertIs(worker.contact_detectors[0].state, ContactState.MOVING)

    def test_motion_above_current_threshold_detects_contact(self):
        events = queue.Queue()
        worker = HandWorker(queue.Queue(), events)
        worker.hand = self.FakeHand()
        worker.hand.currents[0] = 300.0
        worker.contact_settings = ContactDetectorSettings(persistence_time_s=0.0)
        worker.contact_detectors = _build_contact_detectors(worker.contact_settings)

        targets = np.zeros(JOINT_COUNT)
        targets[0] = -0.10
        worker._handle_command("targets", targets)
        worker._control_tick()
        events.get_nowait()
        worker._control_tick()
        telemetry = events.get_nowait().payload

        self.assertEqual(telemetry.contacts, (True,) + (False,) * (JOINT_COUNT - 1))
        self.assertNotEqual(telemetry.contact_feedback[0], 0.0)

    def test_stationary_tracking_error_becomes_contact(self):
        events = queue.Queue()
        worker = HandWorker(queue.Queue(), events)
        worker.hand = self.FakeHand()
        worker.targets = np.zeros(JOINT_COUNT)
        worker.targets[0] = -0.10
        worker.contact_detectors = tuple(
            ContactStateMachine(
                settle_time_s=0.0,
                persistence_time_s=0.0,
                min_engaged_time_s=0.0,
                exit_persistence_time_s=0.0,
            )
            for _ in range(JOINT_COUNT)
        )

        worker._control_tick()
        first = events.get_nowait().payload
        worker._control_tick()
        second = events.get_nowait().payload
        worker._control_tick()
        third = events.get_nowait().payload

        self.assertEqual(first.contacts, (False,) * JOINT_COUNT)
        self.assertEqual(second.contacts, (False,) * JOINT_COUNT)
        self.assertEqual(
            third.contacts, (True,) + (False,) * (JOINT_COUNT - 1)
        )

    def test_commands_and_telemetry_are_mapped_by_motor_id(self):
        events = queue.Queue()
        worker = HandWorker(queue.Queue(), events)
        hand = self.FakeHand()
        # HandConfig.subset preserves YAML motor order while the UI groups each
        # finger as MCP, PIP, then AB/AD.
        worker.ui_to_hand_indices = (1, 0, 2, 4, 3, 5, 7, 6, 8)
        worker.hand = hand
        worker.targets = np.asarray(
            [-0.10, -0.20, 0.01, -0.30, -0.40, 0.02, -0.50, -0.60, 0.03]
        )
        hand.positions = np.asarray(
            [-0.21, -0.11, 0.01, -0.41, -0.31, 0.02, -0.61, -0.51, 0.03]
        )

        worker._control_tick()
        telemetry = events.get_nowait().payload

        np.testing.assert_allclose(
            hand.last_targets,
            (-0.20, -0.10, 0.01, -0.40, -0.30, 0.02, -0.60, -0.50, 0.03),
        )
        self.assertEqual(
            telemetry.positions,
            (-0.11, -0.21, 0.01, -0.31, -0.41, 0.02, -0.51, -0.61, 0.03),
        )


class StreamSequenceTests(unittest.TestCase):
    def test_stream_moves_open_to_close_and_back_to_open(self):
        duration_s = 2.0

        start, speeds, start_complete = _stream_sample(0.0, duration_s)
        closed, _, closed_complete = _stream_sample(duration_s, duration_s)
        opened, _, opened_complete = _stream_sample(2.0 * duration_s, duration_s)
        finished, stopped_speeds, finished_complete = _stream_sample(
            2.1 * duration_s, duration_s
        )
        open_targets = np.full(JOINT_COUNT, STREAM_OPEN_ANGLE_RAD)
        close_targets = np.zeros(JOINT_COUNT)
        close_targets[list(FLEXION_INDICES)] = STREAM_CLOSE_ANGLE_RAD

        np.testing.assert_allclose(start, open_targets)
        np.testing.assert_allclose(closed, close_targets)
        np.testing.assert_allclose(opened, open_targets)
        np.testing.assert_allclose(finished, open_targets)
        np.testing.assert_allclose(
            speeds,
            np.abs(np.asarray(close_targets) - np.asarray(open_targets)) / duration_s,
        )
        self.assertFalse(start_complete)
        self.assertFalse(closed_complete)
        self.assertFalse(opened_complete)
        self.assertTrue(finished_complete)
        np.testing.assert_array_equal(stopped_speeds, np.zeros(JOINT_COUNT))
        np.testing.assert_array_equal(start[list(AB_AD_INDICES)], np.zeros(3))
        np.testing.assert_array_equal(closed[list(AB_AD_INDICES)], np.zeros(3))

    def test_stream_rejects_non_positive_duration(self):
        for duration_s in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(duration_s=duration_s):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    _stream_sample(0.0, duration_s)


class JointLimitTests(unittest.TestCase):
    def test_ui_order_maps_all_nine_motor_ids(self):
        self.assertEqual(
            _ui_order_indices(list(range(4, 13))),
            (1, 0, 2, 4, 3, 5, 7, 6, 8),
        )

    def test_zero_defaults_are_validated_against_configured_limits(self):
        targets = _validated_default_targets((-1.5,) * JOINT_COUNT, (0.5,) * JOINT_COUNT)

        np.testing.assert_array_equal(targets, np.zeros(JOINT_COUNT))

    def test_zero_default_outside_joint_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "PIP"):
            _validated_default_targets(
                (-1.5, 0.1) + (-1.5,) * (JOINT_COUNT - 2),
                (0.5,) * JOINT_COUNT,
            )

    def test_duplicate_ab_ad_motor_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be different"):
            _ui_order_indices(list(range(4, 13)), 5, 4, 5)

    def test_motor_set_covers_exactly_the_three_non_thumb_fingers(self):
        self.assertEqual(set(UI_MOTOR_IDS), set(range(4, 13)))


if __name__ == "__main__":
    unittest.main()
