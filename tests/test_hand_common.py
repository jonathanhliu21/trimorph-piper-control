import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from collection.hand_common import load_hand_config, verify_connected_hand


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


@dataclass(frozen=True)
class _LoadedConfig:
    motor_model: str = "xl330-m077-t"
    motor_ids: tuple[int, ...] = (4, 5)
    port: str | None = None


class _HandConfig:
    @staticmethod
    def load(path):
        return _LoadedConfig()


class LoadHandConfigTests(unittest.TestCase):
    def test_safety_check_uses_active_motor_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hand.yaml"
            path.write_text("motor_ids: [4, 5]\n", encoding="utf-8")
            fake_module = SimpleNamespace(HandConfig=_HandConfig)
            with patch.dict(sys.modules, {"midas_hand_api": fake_module}):
                config = load_hand_config(path, "/dev/hand", [4, 5])
                self.assertEqual(config.motor_ids, (4, 5))
                with self.assertRaisesRegex(RuntimeError, "active motor IDs"):
                    load_hand_config(path, "/dev/hand", list(range(13)))


if __name__ == "__main__":
    unittest.main()
