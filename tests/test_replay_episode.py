import csv
import tempfile
import unittest
from dataclasses import make_dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from collection.thumb_retargeting import ThumbRetargetModel
from replay_episode import (
    CURRENT_TELEMETRY_HZ,
    ReplayCurrentRecorder,
    advance_current_sample_schedule,
    build_replay_config,
    build_replay_hand_config,
    config_values_for_motor_ids,
    load_hand_stream,
    load_piper_stream,
    replay_interval,
    resolve_leader_hand_path,
    sample_stream,
    scale_hand_stream,
    parse_args,
    output_current_csv,
    prepare_hand_stream_for_replay,
)
from scripts.trimorph_api_test.gen_graphs_from_csvs import _load_telemetry


class ReplayEpisodeTests(unittest.TestCase):
    def test_replay_uses_hand_only_collection_config(self):
        with patch(
            "sys.argv",
            [
                "replay_episode.py",
                "--dataset-root",
                "dataset",
                "--hand-config",
                "midas.yaml",
                "--hand-port",
                "/dev/midas",
            ],
        ):
            config = build_replay_config(parse_args(), (4, 5, 12))

        self.assertEqual(config.dataset_root, Path("dataset"))
        self.assertEqual(config.hand_config, Path("midas.yaml"))
        self.assertEqual(config.hand_port, "/dev/midas")
        self.assertEqual(config.active_motors, [4, 5, 12])
        self.assertEqual(config.follower_joint_command_scales[4], 0.4)
        self.assertEqual(config.follower_joint_command_offsets[12], 0.262)

    def test_hand_only_offsets_are_resolved_in_recorded_motor_order(self):
        values = config_values_for_motor_ids(
            {4: 0.1, 12: 0.262}, (12, 5, 4), default=0.0
        )

        np.testing.assert_allclose(values, [0.262, 0.0, 0.1])

    def test_current_sampling_is_downsampled_to_ten_hz(self):
        replay_times = np.arange(31, dtype=np.float64) / 30.0
        next_sample_s = 0.0
        sampled_times = []
        for elapsed_s in replay_times:
            due, next_sample_s = advance_current_sample_schedule(
                float(elapsed_s),
                next_sample_s,
                1.0 / CURRENT_TELEMETRY_HZ,
            )
            if due:
                sampled_times.append(elapsed_s)

        np.testing.assert_allclose(sampled_times, np.arange(11) / 10.0)

    def test_output_current_cli_option_is_opt_in_and_requires_a_path(self):
        with patch("sys.argv", ["replay_episode.py"]):
            self.assertIsNone(parse_args().output_current)
        with patch(
            "sys.argv", ["replay_episode.py", "--output-current", "telemetry.csv"]
        ):
            self.assertEqual(parse_args().output_current, Path("telemetry.csv"))

    def test_current_output_is_compatible_with_graph_generator(self):
        class FakeHand:
            def __init__(self):
                self.sample = 0

            def read_pos_vel_cur(self):
                self.sample += 1
                return (
                    np.asarray([0.1, 0.2]) * self.sample,
                    np.asarray([1.0, 2.0]),
                    np.asarray([10.0, 20.0]) * self.sample,
                )

        recorder = ReplayCurrentRecorder((4, 9))
        hand = FakeHand()
        recorder.sample(hand, np.asarray([0.0, 0.1]), elapsed_s=0.0)
        recorder.sample(hand, np.asarray([0.2, 0.5]), elapsed_s=0.1)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "replay.csv"
            output_current_csv(recorder, output)
            trials, times, motors = _load_telemetry(output)

        np.testing.assert_array_equal(trials, [1, 1])
        np.testing.assert_allclose(times, [0.0, 0.1])
        np.testing.assert_allclose(motors[4]["current"], [10.0, 20.0])
        np.testing.assert_allclose(motors[9]["target"], [0.1, 0.5])
        np.testing.assert_allclose(motors[9]["target_velocity"], [0.0, 4.0])
        np.testing.assert_allclose(motors[9]["actual_velocity"], [2.0, 2.0])

    def test_use_follower_cli_option(self):
        with patch("sys.argv", ["replay_episode.py", "--use-follower"]):
            self.assertTrue(parse_args().use_follower)

    def test_follower_config_is_subset_to_recorded_motor_ids(self):
        ReplayConfig = make_dataclass("ReplayConfig", [("operating_mode", int)])
        loaded = ReplayConfig(operating_mode=5)
        path = Path("midas.yaml")
        with patch(
            "replay_episode.load_follower_config", return_value=loaded
        ) as loader:
            actual = build_replay_hand_config(
                path,
                "/dev/midas",
                (0, 1, 2, 3),
                use_follower=True,
            )

        self.assertIsNot(actual, loaded)
        self.assertEqual(actual.operating_mode, 3)
        self.assertEqual(loaded.operating_mode, 5)
        loader.assert_called_once_with(path, "/dev/midas", (0, 1, 2, 3))

    def test_follower_scale_is_applied_to_all_hand_positions(self):
        times = np.asarray([0.0, 1.0])
        positions = np.asarray([[1.0, -2.0], [3.0, -4.0]])
        scaled_times, scaled_positions = scale_hand_stream(
            (times, positions), 0.25
        )
        np.testing.assert_array_equal(scaled_times, times)
        np.testing.assert_allclose(
            scaled_positions, [[0.25, -0.5], [0.75, -1.0]]
        )
        np.testing.assert_array_equal(positions, [[1.0, -2.0], [3.0, -4.0]])

    def test_joint_specific_scales_are_applied_by_column(self):
        times = np.asarray([0.0, 1.0])
        positions = np.asarray([[1.0, -2.0], [3.0, -4.0]])

        _, scaled_positions = scale_hand_stream(
            (times, positions), np.asarray([0.5, 2.0])
        )

        np.testing.assert_allclose(scaled_positions, [[0.5, -4.0], [1.5, -8.0]])

    def test_midas_replay_retargets_only_joints_two_and_three(self):
        times = np.asarray([0.0, 1.0])
        positions = np.asarray(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]]
        )
        model = ThumbRetargetModel(np.eye(2) * 2.0, np.ones(2))

        _, targets = prepare_hand_stream_for_replay(
            (times, positions),
            (0, 1, 2, 3, 4),
            use_follower=True,
            command_scale=0.5,
            joint_command_scales=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
            thumb_model=model,
        )

        np.testing.assert_allclose(
            targets,
            [[0.05, 0.2, 7.0, 9.0, 1.25], [0.1, 0.3, 9.0, 11.0, 1.5]],
        )

    def test_trimorph_replay_keeps_raw_joint_positions(self):
        times = np.asarray([0.0, 1.0])
        positions = np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        stream = (times, positions)

        result = prepare_hand_stream_for_replay(
            stream,
            (0, 1, 2, 3),
            use_follower=False,
            command_scale=100.0,
            joint_command_scales=np.full(4, 100.0),
            thumb_model=ThumbRetargetModel(np.eye(2) * 100.0, np.ones(2)),
        )

        self.assertIs(result, stream)
        np.testing.assert_array_equal(result[1], positions)

    def test_midas_replay_applies_joint_offsets(self):
        _, targets = prepare_hand_stream_for_replay(
            (np.asarray([0.0]), np.asarray([[0.0, 1.0]])),
            (12, 4),
            use_follower=True,
            command_scale=1.0,
            joint_command_scales=np.asarray([1.0, 0.4]),
            joint_command_offsets=np.asarray([0.262, 0.0]),
            thumb_model=None,
        )

        np.testing.assert_allclose(targets, [[0.262, 0.4]])

    def test_legacy_proprioception_filename_is_still_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode_dir = Path(temp_dir)
            legacy_path = episode_dir / "proprioception.csv"
            legacy_path.touch()
            self.assertEqual(resolve_leader_hand_path(episode_dir), legacy_path)

    def test_hand_stream_is_sorted_and_interpolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leader_hand.csv"
            fields = ["elapsed_s", *(f"pos_{motor_id}" for motor_id in range(13))]
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for elapsed_s, value in ((1.0, 2.0), (0.0, 0.0)):
                    row = {"elapsed_s": elapsed_s}
                    row.update({f"pos_{motor_id}": value for motor_id in range(13)})
                    writer.writerow(row)
            stream = load_hand_stream(path)
            np.testing.assert_allclose(sample_stream(*stream, 0.25), np.full(13, 0.5))

    def test_hand_stream_uses_only_recorded_motor_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "leader_hand.csv"
            with path.open("w", newline="") as output:
                writer = csv.DictWriter(
                    output, fieldnames=["elapsed_s", "pos_4", "pos_5"]
                )
                writer.writeheader()
                writer.writerow({"elapsed_s": 0.0, "pos_4": 1.0, "pos_5": 2.0})
            times, values = load_hand_stream(path)
            np.testing.assert_allclose(times, [0.0])
            np.testing.assert_allclose(values, [[1.0, 2.0]])

    def test_replay_interval_uses_stream_overlap(self):
        hand = (np.asarray([0.1, 2.0]), np.zeros((2, 13)))
        arm = (np.asarray([0.2, 3.0]), np.zeros((2, 6)))
        self.assertEqual(replay_interval(hand, arm), (0.2, 2.0))

    def test_empty_piper_stream_is_allowed_when_requested(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "piper.csv"
            path.write_text("", encoding="utf-8")
            self.assertIsNone(load_piper_stream(path, allow_empty=True))
            with self.assertRaisesRegex(ValueError, "No samples"):
                load_piper_stream(path)


if __name__ == "__main__":
    unittest.main()
