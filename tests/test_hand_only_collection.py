import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from collect_hand_only import (
    DATA_COLLECTION_MODE,
    DEFAULT_FOLLOWER_MEASURE_RATE_HZ,
    DEFAULT_FORCE_FEEDBACK_RATE_HZ,
    DEFAULT_HAND_MEASURE_RATE_HZ,
    FORCE_FEEDBACK_FULL_SCALE_ERROR_RAD,
    FORCE_FEEDBACK_MAX_CURRENT_MA,
    RETARGET_ONLY_MODE,
    TACTILE_FORCE_FEEDBACK_CURRENT_MA,
    HandOnlyCollectionConfig,
    main,
    parse_args,
    run_hand_only,
    start_episode,
    stop_episode,
    worker_specs,
)
from collection.episode import RecordingManager
from collection.config import TRIMORPH_MOTOR_IDS


class HandOnlyCollectionTests(unittest.TestCase):
    def test_bus_rates_default_to_reduced_values(self):
        config = HandOnlyCollectionConfig(
            dataset_root=Path("data_trimorph"),
            hand_config=Path("config/hand.yaml"),
            hand_port="/dev/hand",
            hand_measure_rate_hz=DEFAULT_HAND_MEASURE_RATE_HZ,
            warmup_time=0.0,
            episode_time=30.0,
        )

        self.assertEqual(config.hand_measure_rate_hz, 50.0)
        self.assertEqual(config.force_feedback_rate_hz, 50.0)
        self.assertEqual(config.follower_measure_rate_hz, 100.0)
        self.assertEqual(DEFAULT_FORCE_FEEDBACK_RATE_HZ, 50.0)
        self.assertEqual(DEFAULT_FOLLOWER_MEASURE_RATE_HZ, 100.0)

    def test_force_feedback_tuning_lives_in_hand_only_config(self):
        config = HandOnlyCollectionConfig(
            dataset_root=Path("data_trimorph"),
            hand_config=Path("config/hand.yaml"),
            hand_port="/dev/hand",
            hand_measure_rate_hz=100.0,
            warmup_time=0.0,
            episode_time=30.0,
        )

        self.assertEqual(
            config.force_feedback_max_current_ma,
            FORCE_FEEDBACK_MAX_CURRENT_MA,
        )
        self.assertEqual(
            config.force_feedback_full_scale_error_rad,
            FORCE_FEEDBACK_FULL_SCALE_ERROR_RAD,
        )
        self.assertEqual(
            config.tactile_force_feedback_current_ma,
            TACTILE_FORCE_FEEDBACK_CURRENT_MA,
        )

    def test_joint_command_scales_default_to_configured_values(self):
        config = HandOnlyCollectionConfig(
            dataset_root=Path("data_trimorph"),
            hand_config=Path("config/hand.yaml"),
            hand_port="/dev/hand",
            hand_measure_rate_hz=100.0,
            warmup_time=0.0,
            episode_time=30.0,
        )

        self.assertEqual(
            config.follower_joint_command_scales,
            {
                motor_id: 0.5 if motor_id in (4, 5) else 1.0
                for motor_id in TRIMORPH_MOTOR_IDS
            },
        )

    def test_joint_command_offsets_default_to_zero_midas_pose(self):
        config = HandOnlyCollectionConfig(
            dataset_root=Path("data_trimorph"),
            hand_config=Path("config/hand.yaml"),
            hand_port="/dev/hand",
            hand_measure_rate_hz=100.0,
            warmup_time=0.0,
            episode_time=30.0,
        )

        self.assertEqual(
            config.follower_joint_command_offsets,
            {motor_id: 0.0 for motor_id in TRIMORPH_MOTOR_IDS},
        )

    def test_cli_modes_default_to_data_collection(self):
        with patch("sys.argv", ["collect_hand_only.py"]):
            self.assertEqual(parse_args().mode, DATA_COLLECTION_MODE)
        with patch("sys.argv", ["collect_hand_only.py", "--retarget-only"]):
            self.assertEqual(parse_args().mode, RETARGET_ONLY_MODE)
        with patch("sys.argv", ["collect_hand_only.py", "--data-collection"]):
            self.assertEqual(parse_args().mode, DATA_COLLECTION_MODE)

    def test_cli_parses_disabled_force_feedback_motors(self):
        with patch(
            "sys.argv",
            [
                "collect_hand_only.py",
                "--disable-force-feedback-motors",
                "8",
                "11",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.disable_force_feedback_motors, [8, 11])

    def test_main_copies_disabled_force_feedback_motors_to_config(self):
        with (
            patch(
                "sys.argv",
                [
                    "collect_hand_only.py",
                    "--disable-force-feedback-motors",
                    "8",
                    "8",
                    "11",
                ],
            ),
            patch("collect_hand_only.run_hand_only", return_value=0) as run,
        ):
            self.assertEqual(main(), 0)

        self.assertEqual(
            run.call_args.args[0].disabled_force_feedback_motor_ids,
            (8, 11),
        )

    def test_disabled_force_feedback_motor_must_be_active(self):
        with patch(
            "sys.argv",
            [
                "collect_hand_only.py",
                "--active-motors",
                "4",
                "5",
                "--disable-force-feedback-motors",
                "8",
            ],
        ):
            with self.assertRaisesRegex(ValueError, "not active: \\[8\\]"):
                main()

    def test_show_ui_flag_is_opt_in(self):
        with patch("sys.argv", ["collect_hand_only.py"]):
            self.assertFalse(parse_args().show_ui)
        with patch("sys.argv", ["collect_hand_only.py", "--show-ui"]):
            self.assertTrue(parse_args().show_ui)

    def test_camera_flag_is_opt_in(self):
        with patch("sys.argv", ["collect_hand_only.py"]):
            self.assertFalse(parse_args().add_camera)
        with patch("sys.argv", ["collect_hand_only.py", "--add-camera"]):
            self.assertTrue(parse_args().add_camera)

    def test_camera_worker_is_conditional(self):
        config = HandOnlyCollectionConfig(
            dataset_root=Path("data_trimorph"),
            hand_config=Path("config/hand.yaml"),
            hand_port="/dev/hand",
            hand_measure_rate_hz=100.0,
            warmup_time=0.0,
            episode_time=30.0,
        )
        self.assertEqual([name for name, _ in worker_specs(config)], ["leader_hand"])
        config = replace(config, add_camera=True)
        self.assertEqual(
            [name for name, _ in worker_specs(config)],
            ["task_camera", "leader_hand"],
        )

    def test_tactile_options_are_opt_in(self):
        with patch("sys.argv", ["collect_hand_only.py"]):
            args = parse_args()
        self.assertFalse(args.add_tactile)
        self.assertIsNone(args.paxini_port)
        self.assertIsNone(args.tactile_debug_log)

        with patch(
            "sys.argv",
            [
                "collect_hand_only.py",
                "--add-tactile",
                "--paxini-port",
                "/dev/paxini",
                "--tactile-debug-log",
                "output/tactile.log",
            ],
        ):
            args = parse_args()
        self.assertTrue(args.add_tactile)
        self.assertEqual(args.paxini_port, "/dev/paxini")
        self.assertEqual(args.tactile_debug_log, Path("output/tactile.log"))

    def test_paxini_port_implicitly_enables_tactile_worker(self):
        with (
            patch(
                "sys.argv",
                [
                    "collect_hand_only.py",
                    "--show-ui",
                    "--follower-config",
                    "config/midas_hand_config.yaml",
                    "--follower-port",
                    "/dev/follower",
                    "--paxini-port",
                    "/dev/paxini",
                ],
            ),
            patch("collect_hand_only.run_hand_only", return_value=0) as run,
        ):
            self.assertEqual(main(), 0)

        config = run.call_args.args[0]
        self.assertTrue(config.add_tactile)
        self.assertEqual(config.paxini_port, "/dev/paxini")

    def test_omitted_paxini_port_does_not_conflict_with_omitted_follower(self):
        with (
            patch("sys.argv", ["collect_hand_only.py"]),
            patch("collect_hand_only.run_hand_only", return_value=0) as run,
        ):
            self.assertEqual(main(), 0)

        self.assertFalse(run.call_args.args[0].add_tactile)

    def test_show_ui_requires_follower_configuration(self):
        with patch("sys.argv", ["collect_hand_only.py", "--show-ui"]):
            with self.assertRaisesRegex(ValueError, "--show-ui requires"):
                main()

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

    def test_camera_metadata_is_populated_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HandOnlyCollectionConfig(
                dataset_root=Path(temp_dir),
                hand_config=Path("config/hand.yaml"),
                hand_port="/dev/hand",
                hand_measure_rate_hz=100.0,
                warmup_time=0.0,
                episode_time=30.0,
                add_camera=True,
            )
            recorder = RecordingManager()
            session = start_episode(config, recorder)
            session.append_camera(
                {"frame_idx": 0, "monotonic_ns": session.start_ns, "elapsed_s": "0"}
            )
            session.video_done.set()
            with patch("collection.episode._hand_metadata", return_value={}):
                stop_episode(recorder, "duration_complete")

            metadata = json.loads(session.paths.metadata_path.read_text())
            self.assertEqual(metadata["camera"]["device"], "oak1_depthai")
            self.assertEqual(metadata["camera"]["brightness"], 0)
            self.assertEqual(metadata["camera"]["anti_banding_hz"], "60")
            self.assertEqual(metadata["counts"]["camera_frames"], 1)

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

    def test_retarget_only_does_not_create_an_episode(self):
        class FinishedThread:
            def join(self, timeout=None):
                return None

        def initialize_and_stop(config, shared, recorder, errors):
            shared.shutdown.set()
            return [FinishedThread()]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = HandOnlyCollectionConfig(
                dataset_root=Path(temp_dir),
                hand_config=Path("config/hand.yaml"),
                hand_port="/dev/hand",
                hand_measure_rate_hz=100.0,
                warmup_time=0.0,
                episode_time=30.0,
                mode=RETARGET_ONLY_MODE,
            )
            with patch(
                "collect_hand_only.initialize_worker", side_effect=initialize_and_stop
            ):
                self.assertEqual(run_hand_only(config), 0)

            self.assertFalse((Path(temp_dir) / "raw").exists())

    def test_ui_close_uses_normal_collection_cleanup(self):
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
                episode_time=30.0,
                show_ui=True,
                follower_hand_config=Path("config/midas_hand_config.yaml"),
                follower_hand_port="/dev/follower",
            )
            with (
                patch("collect_hand_only.initialize_worker", return_value=FinishedThread()),
                patch(
                    "collect_hand_only.run_follower_contact_ui",
                    return_value="ui_closed",
                ) as run_ui,
                patch("collection.episode._hand_metadata", return_value={}),
            ):
                self.assertEqual(run_hand_only(config), 0)

            run_ui.assert_called_once()
            metadata = json.loads(
                (Path(temp_dir) / "raw" / "episode_000000" / "metadata.json").read_text()
            )
            self.assertEqual(metadata["stop_reason"], "ui_closed")


if __name__ == "__main__":
    unittest.main()
