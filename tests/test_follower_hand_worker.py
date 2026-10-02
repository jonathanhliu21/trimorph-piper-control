import unittest

import numpy as np

from collection.contact_detection import (
    CONTACT_EXIT_THRESHOLD,
    CONTACT_THRESHOLD,
)
from collection.config import (
    FOLLOWER_PROFILE_ACCELERATION_RAD_S2,
    FOLLOWER_PROFILE_VELOCITY_RAD_S,
)
from collection.follower_hand_worker import (
    adjusted_follower_position_error,
    follower_contact_detection_ready,
    prepare_follower_hand,
)


class PrepareFollowerHandTests(unittest.TestCase):
    def test_applies_shared_replay_motion_profile(self):
        class FakeHand:
            def __init__(self):
                self.configure_calls = []
                self.profile_calls = []

            def configure(self, *, enable_torque):
                self.configure_calls.append(enable_torque)

            def set_motion_profile(self, **kwargs):
                self.profile_calls.append(kwargs)

        hand = FakeHand()
        prepare_follower_hand(hand, configure_hardware=True)

        self.assertEqual(hand.configure_calls, [False])
        self.assertEqual(
            hand.profile_calls,
            [
                {
                    "profile_velocity_rad_s": FOLLOWER_PROFILE_VELOCITY_RAD_S,
                    "profile_acceleration_rad_s2": (
                        FOLLOWER_PROFILE_ACCELERATION_RAD_S2
                    ),
                }
            ],
        )

    def test_contact_error_uses_exit_threshold_for_motors_4_and_5_only(self):
        adjusted = adjusted_follower_position_error(
            motor_ids=(4, 5, 6),
            target_positions=[0.2, -0.2, 1.0],
            measured_positions=[0.0, 0.0, 0.0],
        )

        np.testing.assert_allclose(
            adjusted,
            [
                0.2 - CONTACT_EXIT_THRESHOLD,
                -0.2 + CONTACT_EXIT_THRESHOLD,
                0.0,
            ],
        )

    def test_contact_error_is_zero_below_exit_threshold(self):
        adjusted = adjusted_follower_position_error(
            motor_ids=(4, 5, 6),
            target_positions=[
                CONTACT_EXIT_THRESHOLD,
                -CONTACT_EXIT_THRESHOLD,
                1.0,
            ],
            measured_positions=[0.0, 0.0, 0.0],
        )

        np.testing.assert_array_equal(adjusted, [0.0, 0.0, 0.0])

    def test_contact_detection_arms_only_after_initial_target_is_reached(self):
        self.assertFalse(
            follower_contact_detection_ready(
                motor_ids=(4, 5, 6),
                target_positions=[-0.5, -0.5, 1.0],
                measured_positions=[0.0, -0.5, 1.0],
            )
        )
        self.assertTrue(
            follower_contact_detection_ready(
                motor_ids=(4, 5, 6),
                target_positions=[-0.5, -0.5, 1.0],
                measured_positions=[
                    -0.5 + CONTACT_THRESHOLD / 2.0,
                    -0.5 - CONTACT_THRESHOLD / 2.0,
                    10.0,
                ],
            )
        )


if __name__ == "__main__":
    unittest.main()
