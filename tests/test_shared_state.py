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
        shared.set_follower_position_error(source, motor_ids=(4, 5, 6))
        source[:] = -1.0

        first = shared.latest_follower_position_error()
        self.assertIsNotNone(first)
        np.testing.assert_array_equal(first, [0.1, -0.2, 0.0])
        first[:] = -2.0
        np.testing.assert_array_equal(
            shared.latest_follower_position_error(),
            [0.1, -0.2, 0.0],
        )
        self.assertEqual(
            shared.latest_follower_position_error_by_motor_id(),
            {4: 0.1, 5: -0.2, 6: 0.0},
        )

    def test_follower_motor_ids_must_align_with_errors(self):
        shared = SharedState()
        with self.assertRaises(ValueError):
            shared.set_follower_position_error([0.1, 0.2], motor_ids=(4,))

    def test_follower_contact_snapshot_is_atomic_and_immutable(self):
        shared = SharedState()
        feedback = np.asarray([0.1, -0.2])
        normalized = np.asarray([2.0, 3.0])
        shared.set_follower_contact_snapshot((5, 4), feedback, normalized)
        feedback[:] = 0.0
        normalized[:] = 0.0

        snapshot = shared.latest_follower_contact_snapshot()

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.motor_ids, (5, 4))
        self.assertEqual(snapshot.contact_feedback, (0.1, -0.2))
        self.assertEqual(snapshot.normalized_joint_difference, (2.0, 3.0))

    def test_follower_contact_snapshot_vectors_must_align(self):
        shared = SharedState()
        with self.assertRaises(ValueError):
            shared.set_follower_contact_snapshot((4, 5), [0.1], [2.0, 3.0])

    def test_tactile_contacts_are_published_as_an_immutable_snapshot(self):
        shared = SharedState()
        contacts = [True, False, True, False]
        shared.set_tactile_contacts(contacts)
        contacts[:] = [False] * 4

        self.assertEqual(
            shared.latest_tactile_contacts(), (True, False, True, False)
        )
        shared.set_tactile_contacts(None)
        self.assertIsNone(shared.latest_tactile_contacts())

    def test_tactile_contacts_require_four_sensors(self):
        shared = SharedState()
        with self.assertRaises(ValueError):
            shared.set_tactile_contacts((True, False))


if __name__ == "__main__":
    unittest.main()
