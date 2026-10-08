import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.trimorph_api_test.gen_graphs_from_csvs import (
    _build_parser,
    _finger_plot_groups,
    _load_telemetry,
)


class GenGraphsFromCsvsTests(unittest.TestCase):
    def _write_csv(self, contents: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "telemetry.csv"
        path.write_text(contents, encoding="utf-8")
        return path

    def test_loads_recorded_target_and_actual_velocities(self):
        path = self._write_csv(
            "trial,elapsed_s,motor_4_current_ma,"
            "motor_4_target_position_rad,motor_4_actual_position_rad,"
            "motor_4_target_velocity_rad_s,motor_4_actual_velocity_rad_s\n"
            "1,0.0,10,0.0,0.0,1.5,1.25\n"
            "1,0.1,11,0.1,0.08,1.5,1.10\n"
        )

        _, _, motors = _load_telemetry(path)

        np.testing.assert_allclose(motors[4]["target_velocity"], [1.5, 1.5])
        np.testing.assert_allclose(motors[4]["actual_velocity"], [1.25, 1.10])

    def test_derives_velocity_for_legacy_csvs(self):
        path = self._write_csv(
            "trial,elapsed_s,motor_4_current_ma,"
            "motor_4_target_position_rad,motor_4_actual_position_rad\n"
            "1,0.0,10,0.0,0.0\n"
            "1,0.5,11,1.0,0.5\n"
            "1,1.0,12,2.0,1.0\n"
        )

        _, _, motors = _load_telemetry(path)

        np.testing.assert_allclose(motors[4]["target_velocity"], [2.0, 2.0, 2.0])
        np.testing.assert_allclose(motors[4]["actual_velocity"], [1.0, 1.0, 1.0])

    def test_parses_overlay_and_joint_filter(self):
        args = _build_parser().parse_args(
            ["first.csv", "second.csv", "--overlay", "--joints", "4", "7"]
        )

        self.assertTrue(args.overlay)
        self.assertEqual(args.joints, [4, 7])

    def test_selected_thumb_joints_are_plotted_in_separate_rows(self):
        groups = _finger_plot_groups({0, 1, 2, 3})

        self.assertEqual(
            [motor_ids for _, motor_ids in groups],
            [(0,), (1,), (2,), (3,)],
        )


if __name__ == "__main__":
    unittest.main()
