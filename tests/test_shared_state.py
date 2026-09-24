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


if __name__ == "__main__":
    unittest.main()
