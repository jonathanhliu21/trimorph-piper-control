import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from collection.config import (
    TRIMORPH_MOTOR_IDS,
    build_config,
    joint_command_offsets_for_motor_ids,
    joint_command_scales_for_motor_ids,
    validate_config,
)


class ConfigTests(unittest.TestCase):
    def test_joint_command_scales_default_to_unity(self):
        config = build_config()

        self.assertEqual(
            config.follower_joint_command_scales,
            {motor_id: 1.0 for motor_id in TRIMORPH_MOTOR_IDS},
        )

    def test_joint_command_scales_follow_motor_id_order(self):
        scales = joint_command_scales_for_motor_ids({4: 0.5, 6: 1.5}, (6, 5, 4))

        self.assertEqual(scales, (1.5, 1.0, 0.5))

    def test_joint_command_offsets_follow_motor_id_order(self):
        offsets = joint_command_offsets_for_motor_ids(
            {4: 0.25, 6: -0.1}, (6, 5, 4)
        )

        self.assertEqual(offsets, (-0.1, 0.0, 0.25))

    def _config(self, directory: Path, *, follower: bool, same_port: bool = False):
        hand = directory / "hand.yaml"
        follower_path = directory / "follower.yaml"
        hand.write_text("motor_model: xl330-m077-t\n", encoding="utf-8")
        follower_path.write_text("motor_model: xl330-m077-t\n", encoding="utf-8")
        return replace(
            build_config(),
            hand_config=hand,
            hand_port="/dev/hand",
            follower_hand_config=follower_path if follower else None,
            follower_hand_port=(
                ("/dev/hand" if same_port else "/dev/follower") if follower else None
            ),
        )

    def test_rejects_same_serial_port(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "must be different"):
                validate_config(
                    self._config(Path(temp_dir), follower=True, same_port=True),
                    require_runtime_modules=False,
                )

    def test_accepts_distinct_explicit_hand_configs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            validate_config(
                self._config(Path(temp_dir), follower=True),
                require_runtime_modules=False,
            )

    def test_accepts_omitted_follower(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            validate_config(
                self._config(Path(temp_dir), follower=False),
                require_runtime_modules=False,
            )

    def test_follower_port_alone_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(Path(temp_dir), follower=False),
                follower_hand_port="/dev/follower",
            )
            with self.assertRaisesRegex(ValueError, "requires --follower-config"):
                validate_config(config, require_runtime_modules=False)

    def test_camera_control_defaults_and_valid_boundaries(self):
        self.assertEqual(build_config().task_camera_brightness, 0)
        self.assertEqual(build_config().task_camera_anti_banding_hz, "60")
        with tempfile.TemporaryDirectory() as temp_dir:
            base = self._config(Path(temp_dir), follower=False)
            for brightness in (-10, 10):
                validate_config(
                    replace(base, task_camera_brightness=brightness),
                    require_runtime_modules=False,
                )

    def test_rejects_invalid_camera_controls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = self._config(Path(temp_dir), follower=False)
            with self.assertRaisesRegex(ValueError, "BRIGHTNESS"):
                validate_config(
                    replace(base, task_camera_brightness=11),
                    require_runtime_modules=False,
                )
            with self.assertRaisesRegex(ValueError, "ANTI_BANDING"):
                validate_config(
                    replace(base, task_camera_anti_banding_hz="invalid"),
                    require_runtime_modules=False,
                )

    def test_accepts_180_degree_camera_flip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(Path(temp_dir), follower=False),
                task_camera_flip="180",
            )
            validate_config(config, require_runtime_modules=False)

if __name__ == "__main__":
    unittest.main()
