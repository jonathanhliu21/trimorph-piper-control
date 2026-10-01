import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from scripts.trimorph_api_test.two_joint_test import (
    CONTACT_THRESHOLD,
    CONTACT_VELOCITY_ERROR_GAIN_S,
    _CurrentRecorder,
    _contact_feedback,
    _dynamic_contact_threshold,
    _build_parser,
    _load_config,
    _output_current_csv,
    main,
)


class TwoJointConfigTests(unittest.TestCase):
    def _load_model(self, motor_model: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "motor_model": motor_model,
                        "motor_ids": [4, 5, 6],
                    }
                ),
                encoding="utf-8",
            )
            return _load_config(path, (4, 5))

    def test_loads_xl330_fixture(self):
        config = self._load_model("xl330-m077-t")

        self.assertEqual(config.motor_model, "xl330-m077-t")
        self.assertEqual(config.motor_ids, (4, 5))
        self.assertEqual(config.expected_model_number, 1190)

    def test_loads_xm335_fixture(self):
        config = self._load_model("xm335-t323-t")

        self.assertEqual(config.motor_model, "xm335-t323-t")
        self.assertEqual(config.motor_ids, (4, 5))
        self.assertEqual(config.expected_model_number, 1710)

    def test_normalizes_supported_model_alias(self):
        config = self._load_model("XM335_T323")

        self.assertEqual(config.motor_model, "xm335-t323-t")

    def test_contact_feedback_is_zero_inside_threshold(self):
        feedback = _contact_feedback(
            [-CONTACT_THRESHOLD, 0.0, CONTACT_THRESHOLD]
        )

        np.testing.assert_array_equal(feedback, [0.0, 0.0, 0.0])

    def test_contact_feedback_is_signed_excess_error(self):
        feedback = _contact_feedback(
            [-CONTACT_THRESHOLD - 0.25, CONTACT_THRESHOLD + 0.5]
        )

        np.testing.assert_allclose(feedback, [-0.25, 0.5])

    def test_dynamic_contact_threshold_increases_with_speed(self):
        thresholds = _dynamic_contact_threshold([-2.0, 0.0, 1.0])

        np.testing.assert_allclose(
            thresholds,
            [
                CONTACT_THRESHOLD + 2.0 * CONTACT_VELOCITY_ERROR_GAIN_S,
                CONTACT_THRESHOLD,
                CONTACT_THRESHOLD + CONTACT_VELOCITY_ERROR_GAIN_S,
            ],
        )

    def test_contact_feedback_accepts_joint_specific_thresholds(self):
        feedback = _contact_feedback([0.25, -0.25], [0.2, 0.3])

        np.testing.assert_allclose(feedback, [0.05, 0.0])

    def test_stream_move_refuses_to_overwrite_current_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "current.csv"
            output.touch()
            stderr = io.StringIO()
            argv = [
                "two_joint_test.py",
                "stream_move",
                "--output-current",
                str(output),
            ]

            with mock.patch("sys.argv", argv), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("output current path already exists", stderr.getvalue())

    def test_stream_move_force_allows_existing_current_output(self):
        args = _build_parser().parse_args(
            ["stream_move", "--output-current", "current.csv", "--force"]
        )

        self.assertTrue(args.force)

    def test_current_recorder_outputs_target_and_actual_velocities(self):
        class FakeHand:
            motor_ids = (4, 5)

            @staticmethod
            def read_pos_vel_cur():
                return (
                    np.asarray([-0.1, -0.2]),
                    np.asarray([-0.3, -0.4]),
                    np.asarray([10.0, 20.0]),
                )

        recorder = _CurrentRecorder(FakeHand(), (4, 5))
        recorder.sample(
            [-0.15, -0.25],
            elapsed_s=0.5,
            target_velocities_rad_s=[-1.0, -2.0],
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "telemetry.csv"
            _output_current_csv(recorder, output)
            with output.open(newline="", encoding="utf-8") as stream:
                [row] = list(csv.DictReader(stream))

        self.assertEqual(row["motor_4_target_velocity_rad_s"], "-1.0")
        self.assertEqual(row["motor_4_actual_velocity_rad_s"], "-0.3")
        self.assertEqual(row["motor_5_target_velocity_rad_s"], "-2.0")
        self.assertEqual(row["motor_5_actual_velocity_rad_s"], "-0.4")


if __name__ == "__main__":
    unittest.main()
