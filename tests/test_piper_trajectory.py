import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from scripts.piper_api.record_trajectory import (
    EEF_FIELDS,
    JOINT_FIELDS,
    joint_limit_warning,
    parse_args as parse_record_args,
    read_joint_limits,
    run as run_record,
    write_trajectory,
)
from scripts.piper_api.piper_api import PiperAPI
from scripts.piper_api.replay_trajectory import (
    command_bounded_joint_step,
    interpolate_joint_step,
    load_trajectory,
    parse_args,
    preposition_joints_linearly,
    resolve_trajectory,
    run,
    sample_pose,
    validate_joint_continuity,
    verify_joint_start,
)


class PiperTrajectoryTest(unittest.TestCase):
    def test_joint_limit_warning_near_and_outside_controller_limits(self):
        low, high = -1.22173, 1.22173
        self.assertIsNone(joint_limit_warning(5, 0.0, low, high))
        near = joint_limit_warning(5, low + math.radians(1), low, high)
        self.assertEqual(near[0], "near minimum")
        outside = joint_limit_warning(5, low - 0.001, low, high)
        self.assertEqual(outside[0], "below minimum")
        self.assertIn("J5", outside[1])

    def test_joint_limit_read_failure_only_disables_that_joint_warning(self):
        api = Mock()
        api.robot.get_joint_angle_vel_limits.side_effect = lambda joint, **_: (
            None if joint == 5 else SimpleNamespace(
                msg=SimpleNamespace(min_angle_limit=-1.0, max_angle_limit=1.0)
            )
        )
        with patch("scripts.piper_api.record_trajectory.sys.stderr"):
            limits = read_joint_limits(api)
        self.assertEqual(len(limits), 5)
        self.assertNotIn(5, limits)

    def test_record_episode_duration_cli(self):
        args = parse_record_args(
            ["--output", "output/example.csv", "--episode-s", "4.5"]
        )

        self.assertEqual(args.episode_s, 4.5)
        self.assertEqual(args.output, Path("output/example.csv"))

    def test_joint_position_alias_enables_joint_recording(self):
        args = parse_record_args(["--output", "output/example.csv", "--joint-position"])
        self.assertTrue(args.joint_level)

    def test_record_episode_duration_must_be_positive(self):
        args = parse_record_args(
            ["--output", "output/example.csv", "--episode-s", "0"]
        )

        with self.assertRaisesRegex(ValueError, "--episode-s"):
            run_record(args)

    def test_piper_api_sets_position_mode(self):
        class MotionMode:
            P = object()

        class Options:
            MOTION_MODE = MotionMode

        class Robot:
            OPTIONS = Options

            def __init__(self):
                self.mode = None

            def set_motion_mode(self, mode):
                self.mode = mode

        robot = Robot()
        api = PiperAPI(robot=robot)

        api.set_position_mode()

        self.assertIs(robot.mode, MotionMode.P)

    def test_recorded_csv_loads_for_replay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "trajectory.csv"
            rows = []
            for index, elapsed_s in enumerate((0.0, 1.0)):
                row = {
                    "sample_idx": index,
                    "monotonic_ns": index * 1_000_000_000,
                    "wall_time_s": elapsed_s,
                    "elapsed_s": elapsed_s,
                    "robot_timestamp_s": elapsed_s,
                    "robot_hz": 200.0,
                    "teach_status": "teaching",
                }
                row.update(
                    {
                        field: index + offset
                        for offset, field in enumerate(EEF_FIELDS)
                    }
                )
                rows.append(row)

            write_trajectory(output, rows, force=False)
            times, poses = load_trajectory(output)

            self.assertEqual(times, [0.0, 1.0])
            self.assertEqual(
                sample_pose(times, poses, 0.5),
                [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            )

    def test_joint_level_recording_loads_for_joint_replay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.csv"
            rows = []
            for index in range(2):
                row = {
                    "elapsed_s": float(index),
                    **dict.fromkeys(EEF_FIELDS, 0.0),
                    **dict.fromkeys(JOINT_FIELDS, float(index)),
                }
                rows.append(row)
            write_trajectory(path, rows, force=False)
            times, joints = load_trajectory(path, joint_level=True)
            self.assertEqual(times, [0.0, 1.0])
            self.assertEqual(sample_pose(times, joints, 0.5), [0.5] * 6)

    def test_joint_level_requires_recorded_joint_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.csv"
            write_trajectory(
                path,
                [{"elapsed_s": 0.0, **dict.fromkeys(EEF_FIELDS, 0.0)}],
                force=False,
            )
            with self.assertRaisesRegex(ValueError, "joint_1"):
                load_trajectory(path, joint_level=True)

    def test_joint_start_must_match_recording(self):
        api = Mock()
        api.get_joint_angles.return_value = [0.0] * 6
        with self.assertRaisesRegex(RuntimeError, "not at the recorded joint start"):
            verify_joint_start(api, [0.2] + [0.0] * 5)

    def test_joint_start_accepts_small_tracking_lag_with_warning(self):
        api = Mock()
        api.get_joint_angles.return_value = [0.0] * 6
        with patch("scripts.piper_api.replay_trajectory.sys.stderr") as stderr:
            verify_joint_start(api, [0.113] + [0.0] * 5)
        self.assertTrue(stderr.write.called)

    def test_joint_replay_rejects_discontinuous_recording(self):
        with self.assertRaisesRegex(ValueError, "Recorded J6 jumps"):
            validate_joint_continuity([[0.0] * 6, [0.0] * 5 + [6.28]])

    def test_interpolation_preserves_joint_line_and_bounds_each_axis(self):
        current = [0.0] * 6
        target = [0.4, -0.2, 0.1, 0.0, 0.0, 0.0]
        self.assertEqual(
            interpolate_joint_step(current, target, 0.02),
            [0.02, -0.01, 0.005, 0.0, 0.0, 0.0],
        )

    def test_bounded_command_uses_measured_angles_not_previous_target(self):
        api = Mock()
        api.get_joint_angles.return_value = [0.0] * 6
        command = command_bounded_joint_step(api, [0.5] * 6, 0.02)
        self.assertEqual(command, [0.02] * 6)
        api.move_joint.assert_called_once_with([0.02] * 6, mode="smooth")

    def test_bounded_command_refuses_missing_feedback(self):
        api = Mock()
        api.get_joint_angles.return_value = None
        with self.assertRaisesRegex(RuntimeError, "feedback unavailable"):
            command_bounded_joint_step(api, [0.5] * 6, 0.02)
        api.move_joint.assert_not_called()

    @patch("scripts.piper_api.replay_trajectory.time.sleep")
    def test_joint_preposition_ramps_without_large_command_steps(self, _sleep):
        api = Mock()
        measured = [0.0] * 6
        sent = []
        api.get_joint_angles.side_effect = lambda: list(measured)

        def move_joint(command, *, mode):
            self.assertEqual(mode, "smooth")
            sent.append(list(command))
            measured[:] = command

        api.move_joint.side_effect = move_joint
        preposition_joints_linearly(
            api, [0.08, -0.04, 0.0, 0.0, 0.0, 0.0],
            duration_s=0.2, max_step_rad=0.02,
        )
        self.assertGreaterEqual(len(sent), 4)
        previous = [0.0] * 6
        for command in sent:
            self.assertLessEqual(
                max(abs(a - b) for a, b in zip(command, previous)),
                0.02 + 1e-9,
            )
            previous = command
        self.assertAlmostEqual(sent[-1][0], 0.08)

    @patch("scripts.piper_api.replay_trajectory.time.sleep")
    def test_joint_preposition_aborts_before_motion_if_time_is_too_short(self, _sleep):
        api = Mock()
        api.get_joint_angles.return_value = [0.0] * 6
        with self.assertRaisesRegex(RuntimeError, "increase --preposition-seconds"):
            preposition_joints_linearly(
                api, [0.8] + [0.0] * 5,
                duration_s=1.0, max_step_rad=0.02,
            )
        api.move_joint.assert_not_called()

    def test_directory_resolution_accepts_collection_piper_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            expected = Path(temp_dir) / "piper.csv"
            with expected.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("elapsed_s", *EEF_FIELDS))
                writer.writeheader()
            self.assertEqual(resolve_trajectory(Path(temp_dir)), expected.resolve())

    def test_non_monotonic_timestamps_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("elapsed_s", *EEF_FIELDS))
                writer.writeheader()
                for elapsed_s in (1.0, 0.0):
                    writer.writerow(
                        {
                            "elapsed_s": elapsed_s,
                            **dict.fromkeys(EEF_FIELDS, 0.0),
                        }
                    )
            with self.assertRaisesRegex(ValueError, "not monotonic"):
                load_trajectory(path)

    @patch("scripts.piper_api.replay_trajectory.time.sleep")
    @patch("scripts.piper_api.replay_trajectory.PiperAPI")
    def test_replay_switches_to_position_mode_before_motion(
        self,
        mock_api_type,
        _mock_sleep,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("elapsed_s", *EEF_FIELDS),
                )
                writer.writeheader()
                writer.writerow(
                    {"elapsed_s": 0.0, **dict.fromkeys(EEF_FIELDS, 0.0)}
                )

            api = mock_api_type.return_value
            api.is_ok.return_value = True
            api.get_fps.return_value = 200.0
            api.enable.return_value = True
            args = parse_args([str(path), "--no-preposition"])

            self.assertEqual(run(args), 0)

            api.set_position_mode.assert_called_once_with()
            self.assertLess(
                api.mock_calls.index(call.set_position_mode()),
                api.mock_calls.index(call.move_eef([0.0] * 6)),
            )

    @patch("scripts.piper_api.replay_trajectory.time.sleep")
    @patch("scripts.piper_api.replay_trajectory.PiperAPI")
    def test_joint_replay_routes_through_smooth_move_j_without_limit_preflight(self, mock_api_type, _mock_sleep):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.csv"
            target = [0.0] * 4 + [1.2, 3.2]
            write_trajectory(
                path,
                [{"elapsed_s": 0.0, **dict.fromkeys(EEF_FIELDS, 0.0),
                  **dict(zip(JOINT_FIELDS, target))}],
                force=False,
            )
            api = mock_api_type.return_value
            api.is_ok.return_value = True
            api.get_fps.return_value = 200.0
            api.enable.return_value = True
            api.get_joint_angles.return_value = target
            args = parse_args([str(path), "--joint-level", "--no-preposition"])
            self.assertEqual(run(args), 0)
            api.move_joint.assert_called_once_with(target, mode="smooth")
            api.robot.get_joint_angle_vel_limits.assert_not_called()
            api.move_eef.assert_not_called()


if __name__ == "__main__":
    unittest.main()
