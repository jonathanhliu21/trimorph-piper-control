import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from collection.config import build_config, validate_config


class ConfigTests(unittest.TestCase):
    def _config(self, directory: Path, *, same_port: bool):
        leader = directory / "leader.yaml"
        follower = directory / "follower.yaml"
        leader.write_text("motor_model: xl330-m077-t\n", encoding="utf-8")
        follower.write_text("motor_model: xl330-m077-t\n", encoding="utf-8")
        return replace(
            build_config(),
            leader_hand_config=leader,
            follower_hand_config=follower,
            leader_hand_port="/dev/leader",
            follower_hand_port="/dev/leader" if same_port else "/dev/follower",
        )

    def test_rejects_same_serial_port(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "must be different"):
                validate_config(
                    self._config(Path(temp_dir), same_port=True),
                    require_runtime_modules=False,
                )

    def test_accepts_distinct_explicit_hand_configs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            validate_config(
                self._config(Path(temp_dir), same_port=False),
                require_runtime_modules=False,
            )


if __name__ == "__main__":
    unittest.main()
