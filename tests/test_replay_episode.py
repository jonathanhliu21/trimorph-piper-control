import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from replay_episode import load_hand_stream, replay_interval, sample_stream


class ReplayEpisodeTests(unittest.TestCase):
    def test_hand_stream_is_sorted_and_interpolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "proprioception.csv"
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

    def test_replay_interval_uses_stream_overlap(self):
        hand = (np.asarray([0.1, 2.0]), np.zeros((2, 13)))
        arm = (np.asarray([0.2, 3.0]), np.zeros((2, 6)))
        self.assertEqual(replay_interval(hand, arm), (0.2, 2.0))


if __name__ == "__main__":
    unittest.main()
