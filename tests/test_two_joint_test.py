import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.trimorph_api_test.two_joint_test import _load_config


class TwoJointConfigTests(unittest.TestCase):
    def _load_model(self, motor_model: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "motor_model": motor_model,
                        "motor_ids": [4, 5, 6],
                    }
                ),
                encoding="utf-8",
            )
            return _load_config(path, (4, 5))

    def test_loads_xl330_fixture(self):
        config = self._load_model("xl330-m077-t")

        self.assertEqual(config.motor_model, "xl330-m077-t")
        self.assertEqual(config.motor_ids, (4, 5))
        self.assertEqual(config.expected_model_number, 1190)

    def test_loads_xm335_fixture(self):
        config = self._load_model("xm335-t323-t")

        self.assertEqual(config.motor_model, "xm335-t323-t")
        self.assertEqual(config.motor_ids, (4, 5))
        self.assertEqual(config.expected_model_number, 1710)

    def test_normalizes_supported_model_alias(self):
        config = self._load_model("XM335_T323")

        self.assertEqual(config.motor_model, "xm335-t323-t")


if __name__ == "__main__":
    unittest.main()
