import unittest

from collection.hand_common import verify_connected_hand


class _Config:
    expected_model_number = 1190


class _Hand:
    motor_ids = tuple(range(13))
    config = _Config()

    def __init__(self, responses):
        self._responses = responses

    def ping(self):
        return dict(self._responses)


class VerifyConnectedHandTests(unittest.TestCase):
    def test_accepts_all_expected_motors(self):
        verify_connected_hand(_Hand({motor_id: 1190 for motor_id in range(13)}), role="test")

    def test_rejects_missing_motor(self):
        with self.assertRaisesRegex(RuntimeError, "missing motor IDs"):
            verify_connected_hand(_Hand({motor_id: 1190 for motor_id in range(12)}), role="test")

    def test_rejects_wrong_model(self):
        responses = {motor_id: 1190 for motor_id in range(13)}
        responses[7] = 1710
        with self.assertRaisesRegex(RuntimeError, "unexpected motor models"):
            verify_connected_hand(_Hand(responses), role="test")


if __name__ == "__main__":
    unittest.main()
