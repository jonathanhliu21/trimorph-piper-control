import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data_collection.collection.hand_common import (
    XC330_M181_T_MODEL_NUMBER,
    XL330_M077_T_MODEL_NUMBER,
    is_compatible_dynamixel_model,
    load_follower_config,
    load_hand_config,
    verify_connected_hand,
)


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

    def test_accepts_mixed_xl330_m077_and_xc330_m181_motors(self):
        responses = {
            motor_id: (
                XL330_M077_T_MODEL_NUMBER
                if motor_id % 2 == 0
                else XC330_M181_T_MODEL_NUMBER
            )
            for motor_id in range(13)
        }

        verify_connected_hand(_Hand(responses), role="test")


class CompatibleDynamixelModelTests(unittest.TestCase):
    def test_mixed_models_are_compatible_in_either_config_direction(self):
        self.assertTrue(
            is_compatible_dynamixel_model(
                XC330_M181_T_MODEL_NUMBER, XL330_M077_T_MODEL_NUMBER
            )
        )
        self.assertTrue(
            is_compatible_dynamixel_model(
                XL330_M077_T_MODEL_NUMBER, XC330_M181_T_MODEL_NUMBER
            )
        )

    def test_unrelated_model_still_has_to_match_exactly(self):
        self.assertFalse(is_compatible_dynamixel_model(1710, 1190))
        self.assertFalse(is_compatible_dynamixel_model(1190, 1710))
        self.assertTrue(is_compatible_dynamixel_model(1710, 1710))


@dataclass(frozen=True)
class _LoadedConfig:
    motor_model: str = "xl330-m077-t"
    motor_ids: tuple[int, ...] = tuple(range(13))
    port: str | None = None

    def subset(self, motor_ids):
        selected = set(motor_ids)
        return _LoadedConfig(
            motor_model=self.motor_model,
            motor_ids=tuple(
                motor_id for motor_id in self.motor_ids if motor_id in selected
            ),
            port=self.port,
        )


class _HandConfig:
    @staticmethod
    def load(path):
        return _LoadedConfig()


class LoadHandConfigTests(unittest.TestCase):
    def test_full_config_is_subset_to_active_motor_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hand.yaml"
            path.write_text("motor_ids: [0, 1, 2, 3, 4, 5]\n", encoding="utf-8")
            fake_module = SimpleNamespace(HandConfig=_HandConfig)
            with patch.dict(sys.modules, {"midas_hand_api": fake_module}):
                config = load_hand_config(path, "/dev/hand", [5, 4])

            self.assertEqual(config.motor_ids, (4, 5))
            self.assertEqual(config.port, "/dev/hand")

    def test_follower_config_is_subset_to_active_motor_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "follower.yaml"
            path.write_text("motor_ids: [0, 1, 2, 3, 4, 5]\n", encoding="utf-8")
            fake_module = SimpleNamespace(HandConfig=_HandConfig)
            with patch.dict(sys.modules, {"midas_hand_api": fake_module}):
                config = load_follower_config(path, "/dev/follower", [4, 5])

            self.assertEqual(config.motor_ids, (4, 5))
            self.assertEqual(config.port, "/dev/follower")

    def test_rejects_active_motor_missing_from_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hand.yaml"
            path.write_text("motor_ids: [0, 1, 2, 3, 4, 5]\n", encoding="utf-8")
            fake_module = SimpleNamespace(HandConfig=_HandConfig)
            with patch.dict(sys.modules, {"midas_hand_api": fake_module}):
                with self.assertRaisesRegex(RuntimeError, "not present"):
                    load_hand_config(path, "/dev/hand", [13])

    def test_rejects_duplicate_active_motors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hand.yaml"
            path.write_text("motor_ids: [0, 1, 2, 3, 4, 5]\n", encoding="utf-8")
            fake_module = SimpleNamespace(HandConfig=_HandConfig)
            with patch.dict(sys.modules, {"midas_hand_api": fake_module}):
                with self.assertRaisesRegex(ValueError, "must be unique"):
                    load_hand_config(path, "/dev/hand", [4, 4])


if __name__ == "__main__":
    unittest.main()
