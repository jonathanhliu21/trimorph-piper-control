import unittest
from dataclasses import replace
from pathlib import Path

from data_collection.collection.config import (
    HandRuntimeConfig,
    HandOnlyCollectionConfig,
    joint_command_offsets_for_motor_ids,
    joint_command_scales_for_motor_ids,
    validate_camera_settings,
)


class ConfigTests(unittest.TestCase):
    def _config(self):
        return HandRuntimeConfig(
            dataset_root=Path("dataset"),
            hand_config=Path("hand.yaml"),
            hand_port="/dev/hand",
            hand_measure_rate_hz=50.0,
        )

    def test_collection_inherits_shared_runtime_defaults(self):
        runtime = self._config()
        collection = HandOnlyCollectionConfig(
            dataset_root=runtime.dataset_root,
            hand_config=runtime.hand_config,
            hand_port=runtime.hand_port,
            hand_measure_rate_hz=runtime.hand_measure_rate_hz,
            warmup_time=0.0,
            episode_time=1.0,
        )
        self.assertEqual(
            collection.follower_joint_command_scales, runtime.follower_joint_command_scales
        )
        self.assertEqual(
            collection.follower_joint_command_offsets, runtime.follower_joint_command_offsets
        )
        self.assertEqual(collection.task_camera_exposure_us, runtime.task_camera_exposure_us)
        collection.follower_joint_command_scales[4] = 9.0
        self.assertEqual(runtime.follower_joint_command_scales[4], 0.6)

    def test_joint_command_scales_follow_motor_id_order(self):
        scales = joint_command_scales_for_motor_ids({4: 0.5, 6: 1.5}, (6, 5, 4))

        self.assertEqual(scales, (1.5, 1.0, 0.5))

    def test_joint_command_offsets_follow_motor_id_order(self):
        offsets = joint_command_offsets_for_motor_ids(
            {4: 0.25, 6: -0.1}, (6, 5, 4)
        )

        self.assertEqual(offsets, (-0.1, 0.0, 0.25))

    def test_camera_control_defaults_and_valid_boundaries(self):
        base = self._config()
        self.assertEqual(base.task_camera_exposure_us, 5829)
        self.assertEqual(base.task_camera_iso, 100)
        self.assertEqual(base.task_camera_white_balance_k, 2634)
        validate_camera_settings(base)
        validate_camera_settings(
            replace(
                base,
                task_camera_exposure_us=1_000_000 // base.task_camera_fps,
                task_camera_iso=1600,
                task_camera_white_balance_k=12000,
            )
        )

    def test_rejects_invalid_camera_controls(self):
        base = self._config()
        with self.assertRaisesRegex(ValueError, "EXPOSURE"):
            validate_camera_settings(replace(base, task_camera_exposure_us=0))
        with self.assertRaisesRegex(ValueError, "EXPOSURE"):
            validate_camera_settings(replace(base, task_camera_exposure_us=40_000))
        with self.assertRaisesRegex(ValueError, "ISO"):
            validate_camera_settings(replace(base, task_camera_iso=1601))
        with self.assertRaisesRegex(ValueError, "WHITE_BALANCE"):
            validate_camera_settings(replace(base, task_camera_white_balance_k=999))

    def test_accepts_180_degree_camera_flip(self):
        config = replace(
            self._config(),
            task_camera_flip="180",
        )
        validate_camera_settings(config)


if __name__ == "__main__":
    unittest.main()
