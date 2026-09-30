import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from replay_episode import (
    load_hand_stream,
    load_piper_stream,
    replay_interval,
    resolve_leader_hand_path,
    sample_stream,
    scale_hand_stream,
    parse_args,
)


class ReplayEpisodeTests(unittest.TestCase):
    def test_use_follower_cli_option(self):
        with patch("sys.argv", ["replay_episode.py", "--use-follower"]):
            self.assertTrue(parse_args().use_follower)

    def test_follower_scale_is_applied_to_all_hand_positions(self):
        times = np.asarray([0.0, 1.0])
        positions = np.asarray([[1.0, -2.0], [3.0, -4.0]])
        scaled_times, scaled_positions = scale_hand_stream(
            (times, positions), 0.25
        )
        np.testing.assert_array_equal(scaled_times, times)
        np.testing.assert_allclose(
            scaled_positions, [[0.25, -0.5], [0.75, -1.0]]
        )
        np.testing.assert_array_equal(positions, [[1.0, -2.0], [3.0, -4.0]])

    def test_joint_specific_scales_are_applied_by_column(self):
        times = np.asarray([0.0, 1.0])
        positions = np.asarray([[1.0, -2.0], [3.0, -4.0]])

        _, scaled_positions = scale_hand_stream(
            (times, positions), np.asarray([0.5, 2.0])
        )

        np.testing.assert_allclose(scaled_positions, [[0.5, -4.0], [1.5, -8.0]])

    def test_legacy_proprioception_filename_is_still_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode_dir = Path(temp_dir)
            legacy_path = episode_dir / "proprioception.csv"
            legacy_path.touch()
            self.assertEqual(resolve_leader_hand_path(episode_dir), legacy_path)

    def test_hand_stream_is_sorted_and_interpolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leader_hand.csv"
            fields = ["elapsed_s", *(f"pos_{motor_id}" for motor_id in range(13))]
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for elapsed_s, value in ((1.0, 2.0), (0.0, 0.0)):
                    row = {"elapsed_s": elapsed_s}
                    row.update({f"pos_{motor_id}": value for motor_id in range(13)})
                    writer.writerow(row)
            stream = load_hand_stream(path)
            np.testing.assert_allclose(sample_stream(*stream, 0.25), np.full(13, 0.5))

    def test_hand_stream_uses_only_recorded_motor_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leader_hand.csv"
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output, fieldnames=["elapsed_s", "pos_4", "pos_5"]
                )
                writer.writeheader()
                writer.writerow({"elapsed_s": 0.0, "pos_4": 1.0, "pos_5": 2.0})
            times, values = load_hand_stream(path)
            np.testing.assert_allclose(times, [0.0])
            np.testing.assert_allclose(values, [[1.0, 2.0]])

    def test_replay_interval_uses_stream_overlap(self):
        hand = (np.asarray([0.1, 2.0]), np.zeros((2, 13)))
        arm = (np.asarray([0.2, 3.0]), np.zeros((2, 6)))
        self.assertEqual(replay_interval(hand, arm), (0.2, 2.0))

    def test_empty_piper_stream_is_allowed_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "piper.csv"
            path.write_text("", encoding="utf-8")
            self.assertIsNone(load_piper_stream(path, allow_empty=True))
            with self.assertRaisesRegex(ValueError, "No samples"):
                load_piper_stream(path)


if __name__ == "__main__":
    unittest.main()
