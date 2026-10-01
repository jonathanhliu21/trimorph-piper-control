import unittest

from collection.contact_detection import ContactState, ContactStateMachine


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

    def test_measured_motion_immediately_clears_contact(self):
        detector = ContactStateMachine(settle_time_s=0.0, persistence_time_s=0.0)
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

        self.assertEqual(feedback, 0.0)
        self.assertIs(detector.state, ContactState.MOVING)

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


if __name__ == "__main__":
    unittest.main()
