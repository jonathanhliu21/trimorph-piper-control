import unittest

from collection.config import (
    FOLLOWER_PROFILE_ACCELERATION_RAD_S2,
    FOLLOWER_PROFILE_VELOCITY_RAD_S,
)
from collection.follower_hand_worker import prepare_follower_hand


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


if __name__ == "__main__":
    unittest.main()
