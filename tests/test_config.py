import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from collection.config import build_config, validate_config


class ConfigTests(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
