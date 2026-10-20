import unittest

from data_collection.collection.contact_detection import ContactState, ContactStateMachine


class ContactStateMachineTests(unittest.TestCase):
    def test_motion_then_settle_then_sustained_contact(self):
        detector = ContactStateMachine(
            max_velocity_rad_s=0.1,
            settle_time_s=0.1,
            persistence_time_s=0.05,
        )

        self.assertEqual(
            detector.update(
                commanded_velocity_rad_s=0.2,
                measured_velocity_rad_s=0.0,
                candidate_feedback=1.0,
                now_s=0.0,
            ),
            0.0,
        )
        self.assertIs(detector.state, ContactState.MOVING)

        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=1.0,
            now_s=0.01,
        )
        self.assertIs(detector.state, ContactState.SUPPRESSED)
        self.assertEqual(
            detector.update(
                commanded_velocity_rad_s=0.0,
                measured_velocity_rad_s=0.0,
                candidate_feedback=1.0,
                now_s=0.12,
            ),
            0.0,
        )
        self.assertIs(detector.state, ContactState.ARMED)
        self.assertEqual(
            detector.update(
                commanded_velocity_rad_s=0.0,
                measured_velocity_rad_s=0.0,
                candidate_feedback=1.0,
                now_s=0.16,
            ),
            0.0,
        )
        self.assertEqual(
            detector.update(
                commanded_velocity_rad_s=0.0,
                measured_velocity_rad_s=0.0,
                candidate_feedback=1.0,
                now_s=0.18,
            ),
            1.0,
        )
        self.assertIs(detector.state, ContactState.CONTACT)

    def test_position_residual_keeps_contact_engaged_during_motion(self):
        detector = ContactStateMachine(
            settle_time_s=0.0,
            persistence_time_s=0.0,
            min_engaged_time_s=0.0,
            exit_persistence_time_s=0.0,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=1.0,
            now_s=0.0,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=1.0,
            now_s=0.01,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=1.0,
            now_s=0.02,
        )
        self.assertIs(detector.state, ContactState.CONTACT)

        feedback = detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=detector.max_velocity_rad_s + 0.01,
            candidate_feedback=1.0,
            now_s=0.03,
        )

        self.assertEqual(feedback, 1.0)
        self.assertIs(detector.state, ContactState.CONTACT)

    def test_sustained_current_enters_contact_during_motion(self):
        detector = ContactStateMachine(
            max_velocity_rad_s=0.1,
            motion_current_threshold_ma=250.0,
            persistence_time_s=0.05,
        )

        for now_s in (0.0, 0.04):
            feedback = detector.update(
                commanded_velocity_rad_s=0.2,
                measured_velocity_rad_s=0.1,
                measured_current_ma=-250.0,
                candidate_feedback=-0.08,
                now_s=now_s,
            )
            self.assertEqual(feedback, 0.0)
            self.assertIs(detector.state, ContactState.MOVING)

        feedback = detector.update(
            commanded_velocity_rad_s=0.2,
            measured_velocity_rad_s=0.1,
            measured_current_ma=-250.0,
            candidate_feedback=-0.08,
            now_s=0.06,
        )

        self.assertEqual(feedback, -0.08)
        self.assertIs(detector.state, ContactState.CONTACT)

    def test_motion_current_below_threshold_does_not_enter_contact(self):
        detector = ContactStateMachine(
            max_velocity_rad_s=0.1,
            motion_current_threshold_ma=250.0,
            persistence_time_s=0.0,
        )

        for now_s in (0.0, 0.1):
            feedback = detector.update(
                commanded_velocity_rad_s=0.2,
                measured_velocity_rad_s=0.1,
                measured_current_ma=249.9,
                candidate_feedback=1.0,
                now_s=now_s,
            )
            self.assertEqual(feedback, 0.0)
            self.assertIs(detector.state, ContactState.MOVING)

    def test_contact_stays_engaged_between_entry_and_exit_thresholds(self):
        detector = ContactStateMachine(
            settle_time_s=0.0,
            persistence_time_s=0.0,
            min_engaged_time_s=0.0,
            exit_persistence_time_s=0.0,
            entry_feedback_threshold=0.02,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.0,
            now_s=0.0,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.03,
            now_s=0.01,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.03,
            now_s=0.02,
        )

        feedback = detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.01,
            now_s=0.03,
        )

        self.assertEqual(feedback, 0.01)
        self.assertIs(detector.state, ContactState.CONTACT)

    def test_contact_exit_requires_minimum_engagement_and_sustained_dropout(self):
        detector = ContactStateMachine(
            settle_time_s=0.0,
            persistence_time_s=0.0,
            min_engaged_time_s=0.1,
            exit_persistence_time_s=0.05,
            entry_feedback_threshold=0.02,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.0,
            now_s=0.0,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.03,
            now_s=0.01,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.03,
            now_s=0.02,
        )

        self.assertEqual(
            detector.update(
                commanded_velocity_rad_s=0.0,
                measured_velocity_rad_s=0.0,
                candidate_feedback=0.0,
                now_s=0.03,
            ),
            0.03,
        )
        self.assertEqual(
            detector.update(
                commanded_velocity_rad_s=0.0,
                measured_velocity_rad_s=0.0,
                candidate_feedback=0.0,
                now_s=0.09,
            ),
            0.03,
        )
        self.assertEqual(
            detector.update(
                commanded_velocity_rad_s=0.0,
                measured_velocity_rad_s=0.0,
                candidate_feedback=0.0,
                now_s=0.13,
            ),
            0.0,
        )
        self.assertIs(detector.state, ContactState.ARMED)

    def test_brief_dropout_resets_exit_debounce(self):
        detector = ContactStateMachine(
            settle_time_s=0.0,
            persistence_time_s=0.0,
            min_engaged_time_s=0.0,
            exit_persistence_time_s=0.05,
            entry_feedback_threshold=0.02,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.0,
            now_s=0.0,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.03,
            now_s=0.01,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.03,
            now_s=0.02,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.0,
            now_s=0.03,
        )

        feedback = detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.01,
            now_s=0.06,
        )

        self.assertEqual(feedback, 0.01)
        self.assertIs(detector.state, ContactState.CONTACT)

    def test_contact_must_be_continuous_and_keep_the_same_sign(self):
        detector = ContactStateMachine(settle_time_s=0.0, persistence_time_s=0.05)
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.0,
            now_s=0.0,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=1.0,
            now_s=0.01,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=-1.0,
            now_s=0.04,
        )

        feedback = detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=-1.0,
            now_s=0.07,
        )

        self.assertEqual(feedback, 0.0)
        self.assertIs(detector.state, ContactState.ARMED)

    def test_threshold_drop_resets_armed_persistence_timer(self):
        detector = ContactStateMachine(settle_time_s=0.0, persistence_time_s=0.05)
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.0,
            now_s=0.0,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=1.0,
            now_s=0.01,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=0.0,
            now_s=0.04,
        )
        detector.update(
            commanded_velocity_rad_s=0.0,
            measured_velocity_rad_s=0.0,
            candidate_feedback=1.0,
            now_s=0.05,
        )

        self.assertEqual(
            detector.update(
                commanded_velocity_rad_s=0.0,
                measured_velocity_rad_s=0.0,
                candidate_feedback=1.0,
                now_s=0.09,
            ),
            0.0,
        )
        self.assertEqual(
            detector.update(
                commanded_velocity_rad_s=0.0,
                measured_velocity_rad_s=0.0,
                candidate_feedback=1.0,
                now_s=0.11,
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
