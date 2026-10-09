import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from scripts.piper_api.record_trajectory import (
    EEF_FIELDS,
    parse_args as parse_record_args,
    run as run_record,
    write_trajectory,
)
from scripts.piper_api.piper_api import PiperAPI
from scripts.piper_api.replay_trajectory import (
    load_trajectory,
    parse_args,
    resolve_trajectory,
    run,
    sample_pose,
)


class PiperTrajectoryTest(unittest.TestCase):
    def test_record_episode_duration_cli(self):
        args = parse_record_args(
            ["--output", "output/example.csv", "--episode-s", "4.5"]
        )

        self.assertEqual(args.episode_s, 4.5)
        self.assertEqual(args.output, Path("output/example.csv"))

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


if __name__ == "__main__":
    unittest.main()
