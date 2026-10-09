import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.replay_corl_workshop.manual_replay import (
    INITIAL_HAND_JOINT_ANGLES_RAD,
    PiperTrajectory,
    TrimorphTrajectory,
    initial_hand_target,
    load_alternating_trajectories,
    load_piper_trajectory,
    load_trajectory_paths,
    load_trimorph_trajectory,
    merge_hand_target,
    parse_args,
    run,
    sample_trajectory,
    validate_trimorph_motor_ids,
    wait_for_fresh_pedal_press,
)


PIPER_FIELDS = ("elapsed_s", "x", "y", "z", "roll", "pitch", "yaw")


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_piper(path: Path, times=(0.0, 1.0)) -> None:
    write_csv(
        path,
        PIPER_FIELDS,
        [
            {
                "elapsed_s": elapsed_s,
                "x": index,
                "y": index + 1,
                "z": index + 2,
                "roll": index + 3,
                "pitch": index + 4,
                "yaw": index + 5,
            }
            for index, elapsed_s in enumerate(times)
        ],
    )


def write_trimorph(path: Path, times=(0.0, 1.0), motor_ids=(0, 1)) -> None:
    fields = ["elapsed_s", *(f"pos_{motor_id}" for motor_id in motor_ids)]
    rows = []
    for index, elapsed_s in enumerate(times):
        row = {"elapsed_s": elapsed_s}
        row.update(
            {
                f"pos_{motor_id}": float(index + motor_id)
                for motor_id in motor_ids
            }
        )
        rows.append(row)
    write_csv(path, fields, rows)


class ManualReplayParsingTests(unittest.TestCase):
    def test_list_ignores_comments_and_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            piper = root / "piper.csv"
            hand = root / "hand.csv"
            write_piper(piper)
            write_trimorph(hand)
            trajectory_list = root / "replay.txt"
            trajectory_list.write_text(
                "# alternating replay\n\npiper.csv\n  hand.csv  \n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_trajectory_paths(trajectory_list),
                [piper.resolve(), hand.resolve()],
            )

    def test_empty_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trajectory_list = Path(temp_dir) / "replay.txt"
            trajectory_list.write_text("\n# nothing\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No trajectory CSV paths"):
                load_trajectory_paths(trajectory_list)

    def test_odd_length_alternating_list_is_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [root / "arm1.csv", root / "hand.csv", root / "arm2.csv"]
            write_piper(paths[0])
            write_trimorph(paths[1])
            write_piper(paths[2])

            trajectories = load_alternating_trajectories(paths)

            self.assertIsInstance(trajectories[0], PiperTrajectory)
            self.assertIsInstance(trajectories[1], TrimorphTrajectory)
            self.assertIsInstance(trajectories[2], PiperTrajectory)

    def test_swapped_schema_is_rejected_at_expected_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hand.csv"
            write_trimorph(path)
            with self.assertRaisesRegex(ValueError, "Invalid Piper CSV at entry 1"):
                load_alternating_trajectories([path])

    def test_empty_malformed_nonfinite_and_nonmonotonic_csvs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            empty = root / "empty.csv"
            empty.write_text(",".join(PIPER_FIELDS) + "\n", encoding="utf-8")
            malformed = root / "malformed.csv"
            write_csv(malformed, ("elapsed_s", "x"), [{"elapsed_s": 0, "x": 1}])
            nonfinite = root / "nonfinite.csv"
            write_piper(nonfinite, times=(float("nan"),))
            nonmonotonic = root / "nonmonotonic.csv"
            write_piper(nonmonotonic, times=(1.0, 0.0))

            cases = (
                (empty, "No samples"),
                (malformed, "Missing Piper columns"),
                (nonfinite, "NaN or Inf"),
                (nonmonotonic, "Non-monotonic"),
            )
            for path, message in cases:
                with self.subTest(path=path.name):
                    with self.assertRaisesRegex(ValueError, message):
                        load_piper_trajectory(path)

    def test_duplicate_and_unknown_trimorph_motor_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            duplicate = root / "duplicate.csv"
            duplicate.write_text(
                "elapsed_s,pos_1,pos_01\n0.0,0.1,0.2\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate Trimorph"):
                load_trimorph_trajectory(duplicate)

            unknown = root / "unknown.csv"
            write_trimorph(unknown, times=(0.0,), motor_ids=(7,))
            trajectory = load_trimorph_trajectory(unknown)
            with self.assertRaisesRegex(ValueError, "not present"):
                validate_trimorph_motor_ids([trajectory], (0, 1, 2))

    def test_invalid_csv_fails_before_hardware_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bad = root / "bad.csv"
            write_trimorph(bad)
            trajectory_list = root / "replay.txt"
            trajectory_list.write_text("bad.csv\n", encoding="utf-8")
            args = parse_args(
                [str(trajectory_list), "--hand-port", "/dev/fake-trimorph"]
            )

            with patch(
                "scripts.replay_corl_workshop.manual_replay.load_hand_config"
            ) as load_config, patch(
                "scripts.replay_corl_workshop.manual_replay.PiperAPI"
            ) as piper_type:
                with self.assertRaisesRegex(ValueError, "Invalid Piper CSV"):
                    run(args)
            load_config.assert_not_called()
            piper_type.assert_not_called()


class ManualReplayBehaviorTests(unittest.TestCase):
    def test_initial_hand_target_uses_motor_id_order(self):
        self.assertEqual(
            INITIAL_HAND_JOINT_ANGLES_RAD,
            {
                0: -0.01,
                1: -0.733864183,
                2: 1.316699251,
                3: 0.715970865,
                4: -0.023689328,
                5: -0.276737908,
                6: -0.082523867,
                7: -0.016019424,
                8: 0.039262134,
                9: -0.095999810,
                10: -0.016019424,
                11: -0.149417502,
                12: 0.053922750,
            },
        )
        np.testing.assert_allclose(
            initial_hand_target((12, 0, 5)),
            [0.053922750, -0.01, -0.276737908],
        )

    def test_interpolation_and_partial_hand_target_preserve_other_joints(self):
        times = np.asarray([0.0, 1.0])
        values = np.asarray([[0.0, 2.0], [2.0, 4.0]])
        np.testing.assert_allclose(sample_trajectory(times, values, 0.25), [0.5, 2.5])

        target = merge_hand_target(
            np.asarray([10.0, 20.0, 30.0]),
            (0, 1, 2),
            (1,),
            (99.0,),
        )
        np.testing.assert_allclose(target, [10.0, 99.0, 30.0])

    @patch("scripts.replay_corl_workshop.manual_replay.time.sleep")
    def test_wait_discards_stale_event_and_requires_fresh_press(self, _sleep):
        stale = SimpleNamespace(message="stale")
        fresh = SimpleNamespace(message="fresh")

        class Pedal:
            def __init__(self):
                self.results = [[stale], [], [fresh]]
                self.calls = 0

            def poll(self):
                self.calls += 1
                return self.results.pop(0)

        pedal = Pedal()
        wait_for_fresh_pedal_press(
            pedal, completed=1, total=1, path=Path("piper.csv")
        )
        self.assertEqual(pedal.calls, 3)

    def test_full_run_alternates_commands_and_acknowledges_final_csv(self):
        event_log = []

        class FakePedal:
            active_device = "/dev/input/fake"

            def __init__(self, *args, **kwargs):
                self.poll_count = 0

            def start(self):
                event_log.append("pedal:start")

            def poll(self):
                self.poll_count += 1
                if self.poll_count % 2:
                    return []
                event_log.append("pedal:press")
                return [SimpleNamespace(message="pressed")]

            def stop(self):
                event_log.append("pedal:stop")

        class FakePiper:
            def __init__(self, *args, **kwargs):
                pass

            def connect(self):
                event_log.append("piper:connect")

            def disconnect(self):
                event_log.append("piper:disconnect")

            def is_ok(self):
                return True

            def get_fps(self):
                return 200.0

            def enable(self):
                event_log.append("piper:enable")
                return True

            def set_position_mode(self):
                event_log.append("piper:position-mode")

            def set_speed_percent(self, value):
                pass

            def move_eef(self, pose):
                event_log.append(f"piper:move:{float(pose[0]):g}")

        class FakeHand:
            motor_ids = (0, 1, 2)
            port = "/dev/fake-trimorph"

            def configure(self, *, enable_torque):
                event_log.append(f"hand:configure:{enable_torque}")

            def set_motion_profile(self, **kwargs):
                event_log.append("hand:profile")

            def read_pos(self):
                return np.asarray([10.0, 20.0, 30.0])

            def clip_positions(self, positions):
                return np.asarray(positions)

            def set_positions(self, positions, *, clip):
                values = ",".join(f"{float(value):g}" for value in positions)
                event_log.append(f"hand:move:{values}")

            def enable_torque(self):
                event_log.append("hand:torque")

            def shutdown(self):
                event_log.append("hand:shutdown")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            arm1, hand_path, arm2 = (
                root / "arm1.csv",
                root / "hand.csv",
                root / "arm2.csv",
            )
            write_piper(arm1, times=(0.0,))
            write_trimorph(hand_path, times=(0.0,), motor_ids=(1,))
            write_piper(arm2, times=(0.0,))
            trajectory_list = root / "replay.txt"
            trajectory_list.write_text(
                "arm1.csv\nhand.csv\narm2.csv\n", encoding="utf-8"
            )
            args = parse_args(
                [
                    str(trajectory_list),
                    "--hand-port",
                    "/dev/fake-trimorph",
                    "--no-preposition",
                    "--preposition-seconds",
                    "0",
                    "--settle-seconds",
                    "0",
                ]
            )
            config = SimpleNamespace(motor_ids=(0, 1, 2))
            hand = FakeHand()

            with patch(
                "scripts.replay_corl_workshop.manual_replay.load_hand_config",
                return_value=config,
            ), patch(
                "scripts.replay_corl_workshop.manual_replay.verify_connected_hand"
            ) as verify, patch(
                "scripts.replay_corl_workshop.manual_replay._create_hand",
                return_value=hand,
            ), patch(
                "scripts.replay_corl_workshop.manual_replay.PiperAPI", FakePiper
            ), patch(
                "scripts.replay_corl_workshop.manual_replay.FootPedalTrigger",
                FakePedal,
            ):
                self.assertEqual(run(args), 0)

        verify.assert_called_once_with(hand, role="TRIMORPH")
        active_sequence = [
            entry
            for entry in event_log
            if entry.startswith(("piper:move", "hand:move", "pedal:press"))
        ]
        self.assertEqual(
            active_sequence,
            [
                "hand:move:10,20,30",  # Initial torque-hold target.
                "pedal:press",  # Startup initialization trigger.
                "hand:move:-0.01,-0.733864,1.3167",
                "pedal:press",  # Separate first-Piper trigger.
                "piper:move:0",
                "pedal:press",
                "hand:move:-0.01,1,1.3167",
                "pedal:press",
                "piper:move:0",
                "pedal:press",  # The final CSV is acknowledged too.
            ],
        )
        self.assertIn("piper:position-mode", event_log)
        self.assertIn("hand:torque", event_log)
        self.assertEqual(
            event_log[-3:],
            ["pedal:stop", "hand:shutdown", "piper:disconnect"],
        )

    def test_keyboard_interrupt_cleans_up_both_devices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            piper_path = root / "piper.csv"
            write_piper(piper_path, times=(0.0,))
            trajectory_list = root / "replay.txt"
            trajectory_list.write_text("piper.csv\n", encoding="utf-8")
            args = parse_args(
                [
                    str(trajectory_list),
                    "--hand-port",
                    "/dev/fake-trimorph",
                    "--no-preposition",
                    "--preposition-seconds",
                    "0",
                    "--settle-seconds",
                    "0",
                ]
            )

            config = SimpleNamespace(motor_ids=(0,))
            pedal = MagicMock(active_device="/dev/input/fake")
            api = MagicMock()
            api.is_ok.return_value = True
            api.get_fps.return_value = 200.0
            api.enable.return_value = True
            hand = MagicMock(motor_ids=(0,), port="/dev/fake-trimorph")
            hand.read_pos.return_value = np.asarray([0.0])
            hand.clip_positions.side_effect = lambda positions: np.asarray(positions)

            with patch(
                "scripts.replay_corl_workshop.manual_replay.load_hand_config",
                return_value=config,
            ), patch(
                "scripts.replay_corl_workshop.manual_replay.verify_connected_hand"
            ), patch(
                "scripts.replay_corl_workshop.manual_replay._create_hand",
                return_value=hand,
            ), patch(
                "scripts.replay_corl_workshop.manual_replay.PiperAPI",
                return_value=api,
            ), patch(
                "scripts.replay_corl_workshop.manual_replay.FootPedalTrigger",
                return_value=pedal,
            ), patch(
                "scripts.replay_corl_workshop.manual_replay.replay_piper",
                side_effect=KeyboardInterrupt,
            ):
                self.assertEqual(run(args), 130)

        pedal.stop.assert_called_once_with()
        hand.shutdown.assert_called_once_with()
        api.disconnect.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
