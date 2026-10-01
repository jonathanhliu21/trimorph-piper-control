import unittest

import numpy as np

from collection.shared_state import SharedState


class SharedStateTests(unittest.TestCase):
    def test_hand_target_is_copied_on_write_and_read(self):
        shared = SharedState()
        source = np.arange(13, dtype=np.float64)
        shared.set_hand_target(source)
        source[:] = -1
        first = shared.latest_hand_target()
        self.assertIsNotNone(first)
        np.testing.assert_array_equal(first, np.arange(13, dtype=np.float64))
        first[:] = -2
        np.testing.assert_array_equal(
            shared.latest_hand_target(), np.arange(13, dtype=np.float64)
        )

    def test_follower_position_error_is_copied_on_write_and_read(self):
        shared = SharedState()
        source = np.asarray([0.1, -0.2, 0.0])
        shared.set_follower_position_error(source)
        source[:] = -1.0

        first = shared.latest_follower_position_error()
        self.assertIsNotNone(first)
        np.testing.assert_array_equal(first, [0.1, -0.2, 0.0])
        first[:] = -2.0
        np.testing.assert_array_equal(
            shared.latest_follower_position_error(),
            [0.1, -0.2, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
