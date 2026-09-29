import unittest
from dataclasses import replace
from pathlib import Path

from collection.config import build_config
from collection.runner import worker_specs


class WorkerSpecsTests(unittest.TestCase):
    def test_follower_is_optional(self):
        config = replace(
            build_config(),
            follower_hand_config=None,
            follower_hand_port=None,
        )
        self.assertEqual(
            [name for name, _target in worker_specs(config)],
            ["task_camera", "piper", "leader_hand"],
        )

    def test_enabled_optional_workers_are_included(self):
        config = replace(
            build_config(),
            follower_hand_config=Path("follower.yaml"),
            follower_hand_port="/dev/follower",
        )
        self.assertEqual(
            [name for name, _target in worker_specs(config)],
            ["task_camera", "piper", "leader_hand", "follower"],
        )


if __name__ == "__main__":
    unittest.main()
