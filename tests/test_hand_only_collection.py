import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from collect_hand_only import (
    HandOnlyCollectionConfig,
    run_hand_only,
    start_episode,
    stop_episode,
)
from collection.episode import RecordingManager


class HandOnlyCollectionTests(unittest.TestCase):
    def test_episode_flushes_empty_camera_and_piper_csvs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HandOnlyCollectionConfig(
                dataset_root=Path(temp_dir),
                hand_config=Path("config/hand.yaml"),
                hand_port="/dev/hand",
                hand_measure_rate_hz=100.0,
                warmup_time=0.0,
                episode_time=30.0,
            )
            recorder = RecordingManager()
            session = start_episode(config, recorder)
            session.append_leader_hand(
                {"sample_idx": 0, "monotonic_ns": session.start_ns, "elapsed_s": "0.0"}
            )

            with patch("collection.episode._hand_metadata", return_value={}):
                saved = stop_episode(recorder, "duration_complete")

            self.assertIs(saved, session)
            self.assertEqual(session.paths.camera_timestamps_path.read_text(), "")
            self.assertEqual(session.paths.piper_path.read_text(), "")
            self.assertTrue(session.paths.leader_hand_path.read_text())
            metadata = json.loads(session.paths.metadata_path.read_text())
            self.assertEqual(metadata["counts"]["camera_frames"], 0)
            self.assertEqual(metadata["counts"]["piper_samples"], 0)
            self.assertEqual(metadata["counts"]["leader_hand_samples"], 1)
            self.assertEqual(metadata["paths"]["leader_hand_csv"], "leader_hand.csv")
            self.assertIsNone(metadata["camera"]["device"])
            self.assertIsNone(metadata["piper"]["mode"])

    def test_run_hand_only_owns_timing_and_episode_lifecycle(self):
        class FinishedThread:
            def join(self, timeout=None):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            config = HandOnlyCollectionConfig(
                dataset_root=Path(temp_dir),
                hand_config=Path("config/hand.yaml"),
                hand_port="/dev/hand",
                hand_measure_rate_hz=100.0,
                warmup_time=0.0,
                episode_time=0.0,
            )
            with (
                patch("collect_hand_only.initialize_worker", return_value=FinishedThread()),
                patch("collection.episode._hand_metadata", return_value={}),
            ):
                self.assertEqual(run_hand_only(config), 0)

            episode_dir = Path(temp_dir) / "raw" / "episode_000000"
            self.assertTrue((episode_dir / "leader_hand.csv").exists())
            self.assertEqual((episode_dir / "camera_timestamps.csv").read_text(), "")
            self.assertEqual((episode_dir / "piper.csv").read_text(), "")


if __name__ == "__main__":
    unittest.main()
