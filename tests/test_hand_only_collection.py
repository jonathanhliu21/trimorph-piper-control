import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from data_collection.collection.config import (
    DATA_COLLECTION_MODE,
    DEFAULT_FOLLOWER_MEASURE_RATE_HZ,
    DEFAULT_FORCE_FEEDBACK_RATE_HZ,
    DEFAULT_HAND_MEASURE_RATE_HZ,
    FORCE_FEEDBACK_FULL_SCALE_ERROR_RAD,
    FORCE_FEEDBACK_MAX_CURRENT_MA,
    RETARGET_ONLY_MODE,
    TACTILE_FORCE_FEEDBACK_CURRENT_MA,
    HandOnlyCollectionConfig,
)
from data_collection.collect_hand_only import (
    main,
    parse_args,
    run_hand_only,
    start_episode,
    stop_episode,
    worker_specs,
)
from data_collection.collection.episode import RecordingManager
from data_collection.collection.config import TRIMORPH_MOTOR_IDS


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
                motor_id: (0.6 if motor_id in (4, 7, 10) else
                           0.5 if motor_id in (5, 8, 11) else 1.0)
                for motor_id in TRIMORPH_MOTOR_IDS
            },
        )

    def test_joint_command_offsets_default_to_calibrated_midas_pose(self):
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
            {motor_id: 0.262 if motor_id == 12 else 0.0 for motor_id in TRIMORPH_MOTOR_IDS},
        )

    def test_cli_modes_default_to_data_collection(self):
        with patch("sys.argv", ["collect_hand_only.py"]):
            self.assertEqual(parse_args().mode, DATA_COLLECTION_MODE)
        with patch("sys.argv", ["collect_hand_only.py", "--retarget-only"]):
            self.assertEqual(parse_args().mode, RETARGET_ONLY_MODE)
        with patch("sys.argv", ["collect_hand_only.py", "--data-collection"]):
            self.assertEqual(parse_args().mode, DATA_COLLECTION_MODE)

    def test_joint_level_piper_settings_are_exposed(self):
        with (
            patch(
                "sys.argv",
                ["collect_hand_only.py", "--use-piper", "--joint-level",
                 "--piper-rate-hz", "300", "--piper-channel", "can1"],
            ),
            patch("data_collection.collect_hand_only.run_hand_only", return_value=0) as run,
        ):
            self.assertEqual(main(), 0)
        config = run.call_args.args[0]
        self.assertTrue(config.piper_joint_level)
        self.assertEqual(config.piper_rate_hz, 300.0)
        self.assertEqual(config.piper_channel, "can1")

    def test_joint_position_alias_enables_piper_joint_recording(self):
        with (
            patch("sys.argv", ["collect_hand_only.py", "--use-piper", "--joint-position"]),
            patch("data_collection.collect_hand_only.run_hand_only", return_value=0) as run,
        ):
            self.assertEqual(main(), 0)
        self.assertTrue(run.call_args.args[0].piper_joint_level)

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
            patch("data_collection.collect_hand_only.run_hand_only", return_value=0) as run,
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

    def test_piper_flag_is_opt_in(self):
        with patch("sys.argv", ["collect_hand_only.py"]):
            self.assertFalse(parse_args().use_piper)
        with patch("sys.argv", ["collect_hand_only.py", "--use-piper"]):
            self.assertTrue(parse_args().use_piper)

    def test_quest_flag_is_opt_in_and_mutually_exclusive_with_piper(self):
        with patch("sys.argv", ["collect_hand_only.py"]):
            self.assertFalse(parse_args().use_quest)
        with patch("sys.argv", ["collect_hand_only.py", "--use-quest"]):
            self.assertTrue(parse_args().use_quest)
        with patch(
            "sys.argv", ["collect_hand_only.py", "--use-piper", "--use-quest"]
        ):
            with self.assertRaises(SystemExit):
                parse_args()

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

    def test_piper_worker_is_conditional(self):
        config = HandOnlyCollectionConfig(
            dataset_root=Path("data_trimorph"),
            hand_config=Path("config/hand.yaml"),
            hand_port="/dev/hand",
            hand_measure_rate_hz=100.0,
            warmup_time=0.0,
            episode_time=30.0,
        )
        self.assertNotIn("piper", [name for name, _ in worker_specs(config)])
        config = replace(config, use_piper=True)
        self.assertEqual(
            [name for name, _ in worker_specs(config)],
            ["piper", "leader_hand"],
        )

    def test_quest_worker_is_conditional(self):
        config = HandOnlyCollectionConfig(
            dataset_root=Path("data_trimorph"),
            hand_config=Path("config/hand.yaml"),
            hand_port="/dev/hand",
            hand_measure_rate_hz=100.0,
            warmup_time=0.0,
            episode_time=30.0,
        )
        self.assertNotIn("quest", [name for name, _ in worker_specs(config)])
        config = replace(config, use_quest=True)
        self.assertEqual(
            [name for name, _ in worker_specs(config)],
            ["quest", "leader_hand"],
        )

    def test_main_copies_quest_flag_and_defaults_to_config(self):
        with (
            patch("sys.argv", ["collect_hand_only.py", "--use-quest"]),
            patch("data_collection.collect_hand_only.run_hand_only", return_value=0) as run,
        ):
            self.assertEqual(main(), 0)

        config = run.call_args.args[0]
        self.assertTrue(config.use_quest)
        self.assertEqual(config.quest_host, "0.0.0.0")
        self.assertEqual(config.quest_port, 5005)
        self.assertEqual(config.quest_startup_timeout_s, 5.0)

    def test_main_copies_piper_flag_to_config(self):
        with (
            patch("sys.argv", ["collect_hand_only.py", "--use-piper"]),
            patch("data_collection.collect_hand_only.run_hand_only", return_value=0) as run,
        ):
            self.assertEqual(main(), 0)

        self.assertTrue(run.call_args.args[0].use_piper)

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
            patch("data_collection.collect_hand_only.run_hand_only", return_value=0) as run,
        ):
            self.assertEqual(main(), 0)

        config = run.call_args.args[0]
        self.assertTrue(config.add_tactile)
        self.assertEqual(config.paxini_port, "/dev/paxini")

    def test_omitted_paxini_port_does_not_conflict_with_omitted_follower(self):
        with (
            patch("sys.argv", ["collect_hand_only.py"]),
            patch("data_collection.collect_hand_only.run_hand_only", return_value=0) as run,
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

            with patch("data_collection.collection.episode._hand_metadata", return_value={}):
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
            self.assertIsNone(metadata["piper"]["target_rate_hz"])

    def test_piper_metadata_is_populated_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HandOnlyCollectionConfig(
                dataset_root=Path(temp_dir),
                hand_config=Path("config/hand.yaml"),
                hand_port="/dev/hand",
                hand_measure_rate_hz=100.0,
                warmup_time=0.0,
                episode_time=30.0,
                use_piper=True,
            )
            recorder = RecordingManager()
            session = start_episode(config, recorder)
            session.append_piper(
                {"sample_idx": 0, "monotonic_ns": session.start_ns, "elapsed_s": "0"}
            )
            with patch("data_collection.collection.episode._hand_metadata", return_value={}):
                stop_episode(recorder, "duration_complete")

            metadata = json.loads(session.paths.metadata_path.read_text())
            self.assertEqual(metadata["piper"]["mode"], "passive_drag_teach_logging")
            self.assertEqual(metadata["piper"]["target_rate_hz"], 250.0)
            self.assertEqual(metadata["counts"]["piper_samples"], 1)

    def test_quest_csv_and_metadata_are_written_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HandOnlyCollectionConfig(
                dataset_root=Path(temp_dir),
                hand_config=Path("config/hand.yaml"),
                hand_port="/dev/hand",
                hand_measure_rate_hz=100.0,
                warmup_time=0.0,
                episode_time=30.0,
                use_quest=True,
            )
            recorder = RecordingManager()
            session = start_episode(config, recorder)
            session.append_quest(
                {
                    "sample_idx": 0,
                    "monotonic_ns": session.start_ns,
                    "wall_time_s": "1.0",
                    "elapsed_s": "0.0",
                    "seq": 1,
                    "px": 0.1,
                    "py": 0.2,
                    "pz": 0.3,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                }
            )
            with patch("data_collection.collection.episode._hand_metadata", return_value={}):
                stop_episode(recorder, "duration_complete")

            self.assertTrue(session.paths.quest_path.read_text())
            metadata = json.loads(session.paths.metadata_path.read_text())
            self.assertEqual(metadata["paths"]["quest_csv"], "quest.csv")
            self.assertEqual(metadata["counts"]["quest_samples"], 1)
            self.assertEqual(metadata["quest"]["mode"], "udp_pose_logging")
            self.assertEqual(metadata["quest"]["bind_host"], "0.0.0.0")
            self.assertEqual(metadata["quest"]["port"], 5005)
            self.assertEqual(metadata["quest"]["startup_timeout_s"], 5.0)
            self.assertEqual(
                metadata["quest"]["raw_fields"],
                ["seq", "px", "py", "pz", "qx", "qy", "qz", "qw"],
            )

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
            with patch("data_collection.collection.episode._hand_metadata", return_value={}):
                stop_episode(recorder, "duration_complete")

            self.assertFalse(session.paths.quest_path.exists())
            metadata = json.loads(session.paths.metadata_path.read_text())
            self.assertIsNone(metadata["paths"]["quest_csv"])
            self.assertEqual(metadata["counts"]["quest_samples"], 0)
            self.assertIsNone(metadata["quest"]["mode"])

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
            with patch("data_collection.collection.episode._hand_metadata", return_value={}):
                stop_episode(recorder, "duration_complete")

            metadata = json.loads(session.paths.metadata_path.read_text())
            self.assertEqual(metadata["camera"]["device"], "oak1_depthai")
            self.assertEqual(metadata["camera"]["brightness"], -1)
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
                patch("data_collection.collect_hand_only.initialize_worker", return_value=FinishedThread()),
                patch("data_collection.collection.episode._hand_metadata", return_value={}),
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
                "data_collection.collect_hand_only.initialize_worker", side_effect=initialize_and_stop
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
                patch("data_collection.collect_hand_only.initialize_worker", return_value=FinishedThread()),
                patch(
                    "data_collection.collect_hand_only.run_follower_contact_ui",
                    return_value="ui_closed",
                ) as run_ui,
                patch("data_collection.collection.episode._hand_metadata", return_value={}),
            ):
                self.assertEqual(run_hand_only(config), 0)

            run_ui.assert_called_once()
            metadata = json.loads(
                (Path(temp_dir) / "raw" / "episode_000000" / "metadata.json").read_text()
            )
            self.assertEqual(metadata["stop_reason"], "ui_closed")


if __name__ == "__main__":
    unittest.main()
