import unittest

import numpy as np

from data_collection.collection.tactile_contact_detection import (
    TactileContactConfig,
    TactileContactDetector,
)


class TactileContactDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = TactileContactDetector(
            TactileContactConfig(
                baseline_window_s=0.20,
                median_window_samples=3,
                entry_threshold_n=2.0,
                release_threshold_n=1.0,
                entry_persistence_s=0.05,
                release_persistence_s=0.05,
                baseline_adaptation_tau_s=10.0,
                min_baseline_samples=3,
            ),
            name="middle",
        )

    def _update(self, total_force_n, now_s):
        # Splitting the total across taxels verifies that only their sum matters.
        values = np.asarray([0.25, 0.75]) * float(total_force_n)
        return self.detector.update(values, now_s=now_s)

    def _learn_baseline(self, total_force_n=9.0):
        for index in range(5):
            self.assertFalse(self._update(total_force_n, index * 0.05))
        self.assertTrue(self.detector.ready)

    def test_learns_nonzero_no_contact_baseline(self):
        self._learn_baseline(9.0)

        self.assertAlmostEqual(self.detector.baseline_force_n, 9.0)
        self.assertAlmostEqual(self.detector.baseline_delta_n, 0.0)
        self.assertFalse(self.detector.contact)

    def test_median_filter_rejects_single_force_spike(self):
        self._learn_baseline()

        self.assertFalse(self._update(20.0, 0.25))
        self.assertAlmostEqual(self.detector.filtered_total_force_n, 9.0)
        self.assertFalse(self._update(9.0, 0.30))
        self.assertFalse(self.detector.contact)

    def test_sustained_force_above_baseline_enters_after_persistence(self):
        self._learn_baseline()

        self.assertFalse(self._update(12.0, 0.25))
        self.assertFalse(self._update(12.0, 0.30))
        self.assertFalse(self._update(12.0, 0.35))
        self.assertTrue(self._update(12.0, 0.40))

        self.assertGreaterEqual(self.detector.baseline_delta_n, 2.0)
        self.assertTrue(self.detector.contact)

    def test_hysteresis_holds_between_entry_and_release_thresholds(self):
        self._learn_baseline()
        for now_s in (0.25, 0.30, 0.35, 0.40):
            self._update(12.0, now_s)
        self.assertTrue(self.detector.contact)

        for now_s in (0.45, 0.50, 0.55, 0.60):
            self.assertTrue(self._update(10.5, now_s))

        self.assertGreater(self.detector.baseline_delta_n, 1.0)
        self.assertLess(self.detector.baseline_delta_n, 2.0)

    def test_release_requires_filtered_force_and_persistence(self):
        self._learn_baseline()
        for now_s in (0.25, 0.30, 0.35, 0.40):
            self._update(12.0, now_s)
        self.assertTrue(self.detector.contact)

        self.assertTrue(self._update(9.0, 0.45))
        self.assertTrue(self._update(9.0, 0.50))
        self.assertFalse(self._update(9.0, 0.55))

    def test_baseline_adapts_slowly_only_while_not_in_contact(self):
        self._learn_baseline()
        initial_baseline = self.detector.baseline_force_n

        for index in range(1, 21):
            self._update(9.5, 0.20 + index * 0.05)

        self.assertGreater(self.detector.baseline_force_n, initial_baseline)
        self.assertLess(self.detector.baseline_force_n, 9.5)
        self.assertFalse(self.detector.contact)

    def test_debug_log_reports_baseline_filter_and_hysteresis(self):
        with self.assertLogs(
            "data_collection.collection.tactile_contact_detection", level="DEBUG"
        ) as captured:
            self._learn_baseline()
            for now_s in (0.25, 0.30, 0.35, 0.40):
                self._update(12.0, now_s)

        output = "\n".join(captured.output)
        self.assertIn("sensor=middle", output)
        self.assertIn("event=baseline_ready", output)
        self.assertIn("event=entry_start", output)
        self.assertIn("event=contact_enter", output)
        self.assertIn("raw_force=", output)
        self.assertIn("median_force=", output)
        self.assertIn("baseline=", output)
        self.assertIn("baseline_delta=", output)


if __name__ == "__main__":
    unittest.main()
