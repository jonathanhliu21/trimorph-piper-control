import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_collection"))

from get_tracking_error import TrackingConfig, annotate_samples, summarize_samples


class TrackingErrorAnalysisTests(unittest.TestCase):
    def _config(self):
        return TrackingConfig(
            leader_config=Path("leader.yaml"),
            leader_port="leader",
            follower_config=Path("follower.yaml"),
            follower_port="follower",
            active_motors=(4,),
            speed_bin_edges_rad_s=(0.5, 1.0),
            near_zero_velocity_rad_s=0.1,
            reversal_exclusion_window_s=0.05,
        )

    @staticmethod
    def _row(elapsed, velocity, error, *, fresh=True):
        return {
            "joint_id": 4,
            "joint_name": "index_pip",
            "elapsed_s": elapsed,
            "opening_sign": 1,
            "issued_velocity_rad_s": velocity,
            "raw_error_rad": error,
            "measurement_valid": True,
            "alignment_valid": True,
            "sample_fresh": fresh,
        }

    def test_signed_median_and_direction_are_preserved(self):
        rows = [
            self._row(0.0, 0.2, -0.3),
            self._row(0.2, 0.2, -0.1),
            self._row(0.4, 0.2, 0.2),
        ]
        annotate_samples(rows, self._config())
        summary = summarize_samples(rows)
        self.assertEqual([row["direction"] for row in rows], ["OPENING"] * 3)
        self.assertAlmostEqual(summary[0]["median_e_free_rad"], -0.1)
        self.assertAlmostEqual(summary[0]["mad_rad"], 0.2)

    def test_near_zero_stale_and_reversal_window_are_excluded(self):
        rows = [
            self._row(0.00, 0.0, 0.0),
            self._row(0.10, 0.3, 0.1, fresh=False),
            self._row(0.20, 0.3, 0.1),
            self._row(0.30, -0.3, -0.1),
            self._row(0.50, -0.3, -0.1),
        ]
        annotate_samples(rows, self._config())
        self.assertIn("near_zero_velocity", rows[0]["exclusion_reason"])
        self.assertIn("stale_sample", rows[1]["exclusion_reason"])
        self.assertIn("reversal_window", rows[3]["exclusion_reason"])
        self.assertTrue(rows[2]["baseline_eligible"])
        self.assertTrue(rows[4]["baseline_eligible"])
        self.assertEqual(rows[4]["direction"], "CLOSING")


if __name__ == "__main__":
    unittest.main()
