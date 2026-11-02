import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.dataset import check_episodes

SECOND = 1_000_000_000


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_episode(root, name, *, quest_moves):
    episode = root / "raw" / name
    episode.mkdir(parents=True)
    start = 10 * SECOND
    (episode / "metadata.json").write_text(json.dumps(dict(
        status="complete", stop_reason="foot_pedal", duration_s=2.0, errors=[],
        started_monotonic_ns=start, stopped_monotonic_ns=start + 2 * SECOND,
        trimorph=dict(measure_rate_hz=50.0))))
    write_csv(episode / "leader_hand.csv", [
        dict(monotonic_ns=start + i * SECOND // 50, **{f"pos_{m}": 0.01 * i * (m + 1) for m in range(13)})
        for i in range(100)])
    write_csv(episode / "quest.csv", [
        dict(monotonic_ns=start + i * SECOND // 72, seq=i, px=0.002 * i if quest_moves else 0.1, py=0.8, pz=1.3,
             qx=0, qy=0, qz=0, qw=1)
        for i in range(144)])


class CheckEpisodesTests(unittest.TestCase):
    def run_check(self, root):
        output = io.StringIO()
        with patch("sys.argv", ["check_episodes.py", str(root)]), contextlib.redirect_stdout(output):
            code = check_episodes.main()
        return code, output.getvalue()

    def test_clean_episode_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            make_episode(Path(directory), "episode_000000", quest_moves=True)
            code, output = self.run_check(Path(directory))
        self.assertEqual(code, 0)
        self.assertIn("episode_000000: OK", output)
        self.assertIn("50.0 Hz", output)

    def test_frozen_quest_and_unsaved_episode_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_episode(root, "episode_000000", quest_moves=False)
            unsaved = root / "raw" / "episode_000001"
            unsaved.mkdir()
            (unsaved / "camera.mp4").write_bytes(b"")
            code, output = self.run_check(root)
        self.assertEqual(code, 1)
        self.assertIn("Quest position frozen", output)
        self.assertIn("episode_000001: no metadata.json", output)


if __name__ == "__main__":
    unittest.main()
