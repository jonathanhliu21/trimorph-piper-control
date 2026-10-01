import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from scripts.trimorph_api_test.measure_from_csvs import (
    ABSOLUTE_ERROR_PERCENTILES,
    main,
    measure_position_mae,
)


class MeasureFromCsvsTests(unittest.TestCase):
    def _write_csv(self, contents: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "telemetry.csv"
        path.write_text(contents, encoding="utf-8")
        return path

    def test_computes_mae_and_absolute_error_percentiles(self):
        path = self._write_csv(
            "motor_4_target_position_rad,motor_4_actual_position_rad\n"
            "0,0\n"
            "1,0\n"
            "2,0\n"
            "3,0\n"
        )

        [result] = measure_position_mae(path)

        self.assertEqual(result.sample_count, 4)
        self.assertAlmostEqual(result.mae_rad, 1.5)
        self.assertEqual(ABSOLUTE_ERROR_PERCENTILES, (50, 90, 95, 99, 100))
        expected = (1.5, 2.7, 2.85, 2.97, 3.0)
        for actual, expected_value in zip(
            result.percentile_errors_rad, expected
        ):
            self.assertAlmostEqual(actual, expected_value)

    def test_cli_keeps_mae_output_and_prints_per_motor_and_overall_percentiles(self):
        path = self._write_csv(
            "motor_4_target_position_rad,motor_4_actual_position_rad,"
            "motor_5_target_position_rad,motor_5_actual_position_rad\n"
            "1,0,2,0\n"
            "3,0,4,0\n"
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exit_code = main([str(path)])

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("motor 4: 2.000000 rad", text)
        self.assertIn("overall: 2.500000 rad", text)
        self.assertEqual(text.count("absolute-error percentiles (rad):"), 3)
        self.assertIn("p50=2.000000", text)
        self.assertIn("p99=3.970000", text)
        self.assertIn("p100=4.000000", text)


if __name__ == "__main__":
    unittest.main()
