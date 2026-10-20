import contextlib
import csv
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from data_collection.collection.contact_detection import (
    CONTACT_THRESHOLD,
    contact_feedback as shared_contact_feedback,
)
from scripts.trimorph_api_test.joint_test import (
    GOTO_JOINTS,
    TEST_FINGER_MOTOR_IDS,
    _CurrentRecorder,
    _build_parser,
    _load_config,
    _motion_angles,
    _output_current_csv,
    _run_home,
    _run_goto,
    _validate_home_joints,
    contact_feedback,
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

    def test_reuses_shared_contact_feedback(self):
        self.assertIs(contact_feedback, shared_contact_feedback)

        feedback = contact_feedback([-CONTACT_THRESHOLD, 0.0, CONTACT_THRESHOLD])

        np.testing.assert_array_equal(feedback, [0.0, 0.0, 0.0])

    def test_contact_feedback_is_signed_excess_error(self):
        feedback = contact_feedback(
            [-CONTACT_THRESHOLD - 0.25, CONTACT_THRESHOLD + 0.5]
        )

        np.testing.assert_allclose(feedback, [-0.25, 0.5])

    def test_contact_feedback_accepts_joint_specific_thresholds(self):
        feedback = contact_feedback([0.25, -0.25], [0.2, 0.3])

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

    def test_test_finger_is_unset_when_omitted(self):
        args = _build_parser().parse_args(["record"])

        self.assertIsNone(args.test_finger)

    def test_home_accepts_joint_selection(self):
        args = _build_parser().parse_args(["home", "--joints", "0", "2"])

        self.assertEqual(args.joints, [0, 2])

    def test_goto_parser_defaults(self):
        args = _build_parser().parse_args(["goto"])

        self.assertEqual(args.command, "goto")
        self.assertEqual(args.timeout_s, 4.0)
        self.assertFalse(args.yes)

    def test_goto_enables_torque_at_recorded_positions(self):
        config = _load_config(
            Path("config/xl330_test_config.yaml"),
        )
        config = replace(config, joint_lower_limits=(-3.14,) * config.n_motors)
        calls = []

        class FakeHand:
            def __init__(self, hand_config, autoconnect=False):
                self.config = hand_config
                self.motor_ids = hand_config.motor_ids
                self.is_connected = False

            def connect(self):
                self.is_connected = True

            def ping(self):
                return {motor_id: 1190 for motor_id in self.motor_ids}

            def configure(self, enable_torque=False):
                calls.append(("configure", enable_torque))

            def read_pos(self):
                return np.zeros(13)

            def set_positions(self, positions, clip=True):
                calls.append(("set_positions", tuple(positions), clip))

            def set_motion_profile(self, **kwargs):
                calls.append(("set_motion_profile", kwargs))

            def enable_torque(self, motor_ids):
                calls.append(("enable_torque", tuple(motor_ids)))

            def shutdown(self):
                calls.append(("shutdown",))
                self.is_connected = False

        args = mock.Mock(
            config=Path("config/xl330_test_config.yaml"),
            timeout_s=4.0,
            yes=True,
        )
        with mock.patch(
            "scripts.trimorph_api_test.joint_test.MidasHand", FakeHand
        ), mock.patch(
            "scripts.trimorph_api_test.joint_test._wait_for_motor"
        ), mock.patch(
            "scripts.trimorph_api_test.joint_test.time.sleep",
            side_effect=KeyboardInterrupt,
        ):
            _run_goto(args, config)

        target = tuple(GOTO_JOINTS.values())
        self.assertIn(("set_positions", target, True), calls)
        self.assertIn(("enable_torque", tuple(range(13))), calls)
        self.assertEqual(calls[-1], ("shutdown",))

    def test_home_joints_must_belong_to_selected_finger(self):
        with self.assertRaisesRegex(ValueError, "do not belong to thumb"):
            _validate_home_joints([0, 4], "thumb")

    def test_home_joints_allow_selected_finger_motor_ids(self):
        _validate_home_joints([0, 2], "thumb")

    def test_home_joints_without_test_finger_allow_full_hand_selection(self):
        _validate_home_joints([0, 12], None)

    def test_load_config_without_motor_ids_keeps_full_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "motor_model": "xl330-m077-t",
                        "motor_ids": list(range(13)),
                    }
                ),
                encoding="utf-8",
            )

            config = _load_config(path)

        self.assertEqual(config.motor_ids, tuple(range(13)))

    def test_joint_home_preserves_unselected_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.yaml"
            original_offsets = {motor_id: float(motor_id) for motor_id in range(13)}
            path.write_text(
                yaml.safe_dump(
                    {
                        "placeholders_unreviewed": False,
                        "motor_model": "xl330-m077-t",
                        "motor_ids": list(range(13)),
                        "home_offsets": original_offsets,
                        "homing": {
                            "thumb": [
                                [motor_id, f"Thumb joint {motor_id}", 0.0, 1]
                                for motor_id in range(4)
                            ],
                            "fingers": [],
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            config = _load_config(path, TEST_FINGER_MOTOR_IDS["thumb"])
            calibrated = replace(
                config,
                home_offsets=tuple(
                    100.0 + motor_id for motor_id in config.motor_ids
                ),
            )

            class FakeHand:
                def __init__(self, hand_config, autoconnect=False):
                    self.config = hand_config
                    self.motor_ids = hand_config.motor_ids
                    self.is_connected = False

                def connect(self):
                    self.is_connected = True

                def ping(self):
                    return {motor_id: 1190 for motor_id in self.motor_ids}

                def configure(self, enable_torque=False):
                    pass

                def shutdown(self):
                    self.is_connected = False

            args = mock.Mock(
                config=path,
                yes=True,
                joints=[0, 2],
                current_threshold_ma=200.0,
                velocity_rad_s=1.5,
            )
            with mock.patch(
                "scripts.trimorph_api_test.joint_test.MidasHand", FakeHand
            ), mock.patch(
                "scripts.trimorph_api_test.joint_test.home_hand",
                return_value=calibrated,
            ):
                _run_home(args, config)

            saved_offsets = yaml.safe_load(
                path.read_text(encoding="utf-8")
            )["home_offsets"]

        self.assertEqual(
            saved_offsets,
            {
                **original_offsets,
                0: 100.0,
                2: 102.0,
            },
        )

    def test_test_finger_accepts_each_motor_group(self):
        parser = _build_parser()

        for test_finger in ("thumb", "finger-1", "finger-2", "finger-3"):
            with self.subTest(test_finger=test_finger):
                args = parser.parse_args(
                    ["--test-finger", test_finger, "record"]
                )
                self.assertEqual(args.test_finger, test_finger)

    def test_test_finger_motor_groups(self):
        self.assertEqual(
            TEST_FINGER_MOTOR_IDS,
            {
                "thumb": (0, 1, 2, 3),
                "all": range(13),
                "finger-1": (4, 5, 6),
                "finger-2": (7, 8, 9),
                "finger-3": (10, 11, 12),
            },
        )

    def test_motion_angles_are_joint_specific(self):
        expected = (
            (0.0, -1.4),
            (0.0, -1.4),
            (0.8, -0.1),
            (0.0, 1.5),
            (0.0, -1.5),
            (0.0, -1.4),
            (0.0, 0.0),
            (0.0, -1.5),
            (0.0, -1.4),
            (0.0, 0.0),
            (0.0, -1.5),
            (0.0, -1.4),
            (0.0, 0.0),
        )

        self.assertEqual(tuple(_motion_angles(i) for i in range(13)), expected)

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
