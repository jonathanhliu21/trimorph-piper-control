import unittest
from dataclasses import replace
from pathlib import Path

from collection.config import build_config
from collection.runner import worker_specs


class WorkerSpecsTests(unittest.TestCase):
    def test_camera_and_follower_are_optional(self):
        config = replace(
            build_config(),
            task_camera_enabled=False,
            follower_hand_config=None,
            follower_hand_port=None,
        )
        self.assertEqual(
            [name for name, _target in worker_specs(config)],
            ["piper", "proprioception"],
        )

    def test_enabled_optional_workers_are_included(self):
        config = replace(
            build_config(),
            task_camera_enabled=True,
            follower_hand_config=Path("follower.yaml"),
            follower_hand_port="/dev/follower",
        )
        self.assertEqual(
            [name for name, _target in worker_specs(config)],
            ["task_camera", "piper", "proprioception", "follower"],
        )


if __name__ == "__main__":
    unittest.main()
