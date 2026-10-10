import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

from collection.piper_worker import piper_workspace_warning, run_piper_worker


class PiperWorkspaceWarningTests(unittest.TestCase):
    def test_normal_state_has_no_warning(self):
        self.assertIsNone(
            piper_workspace_warning(
                {"arm_status": "NORMAL(0x0)", "err_code": 0}
            )
        )

    def test_target_limit_is_reported(self):
        warning = piper_workspace_warning(
            {"arm_status": "TARGET_POS_EXCEEDS_LIMIT(0x4)", "err_code": 0}
        )

        self.assertIsNotNone(warning)
        self.assertIn("workspace/kinematic limit", warning)
        self.assertIn("TARGET_POS_EXCEEDS_LIMIT", warning)

    def test_singularity_and_joint_limit_bits_are_reported(self):
        warning = piper_workspace_warning(
            {
                "arm_status": "SINGULARITY_POINT(0x3)",
                "err_code": (1 << 8) | (1 << 12),
            }
        )

        self.assertIsNotNone(warning)
        self.assertIn("SINGULARITY_POINT", warning)
        self.assertIn("joint_angle_limit=[1, 5]", warning)

    def test_joint_level_worker_records_joints_without_commanding_arm(self):
        shutdown = threading.Event()
        shared = SimpleNamespace(shutdown=shutdown)
        rows = []
        session = SimpleNamespace(
            elapsed_s=lambda _sample_ns: 0.0,
            append_piper=rows.append,
        )
        recorder = SimpleNamespace(current=lambda: session)
        config = SimpleNamespace(
            piper_firmware="v183",
            piper_interface="socketcan",
            piper_channel="can0",
            piper_installation_pos="right",
            piper_payload="half",
            piper_rate_hz=250.0,
            teacher_mode_poll_hz=20.0,
            piper_joint_level=True,
        )
        init_done = threading.Event()
        with patch("scripts.piper_api.piper_api.PiperAPI") as api_type:
            api = api_type.return_value
            api.is_ok.return_value = True
            api.get_fps.return_value = 200.0
            api.get_state.return_value = {
                "teach_status": "START_RECORDING", "arm_status": "NORMAL", "err_code": 0
            }
            api.robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
                msg=SimpleNamespace(min_angle_limit=-1.0, max_angle_limit=1.0)
            )

            def eef_once():
                shutdown.set()
                return SimpleNamespace(pose=[0.1] * 6, timestamp=1.0, hz=200.0)

            api.get_eef_state.side_effect = eef_once
            api.get_joint_state.return_value = SimpleNamespace(joints=[0.2] * 6)
            run_piper_worker(config, shared, recorder, init_done=init_done)
            api.enable.assert_not_called()
            api.move_eef.assert_not_called()
            api.move_joint.assert_not_called()

        self.assertTrue(init_done.is_set())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["joint_1"], "0.200000000")
        self.assertEqual(rows[0]["joint_6"], "0.200000000")

    def test_joint_level_worker_warns_when_measured_joint_exceeds_limit(self):
        shutdown = threading.Event()
        shared = SimpleNamespace(shutdown=shutdown)
        rows = []
        session = SimpleNamespace(elapsed_s=lambda _sample_ns: 0.0, append_piper=rows.append)
        recorder = SimpleNamespace(current=lambda: session)
        config = SimpleNamespace(
            piper_firmware="v183",
            piper_interface="socketcan",
            piper_channel="can0",
            piper_installation_pos="right",
            piper_payload="half",
            piper_rate_hz=250.0,
            teacher_mode_poll_hz=20.0,
            piper_joint_level=True,
        )
        with (
            patch("scripts.piper_api.piper_api.PiperAPI") as api_type,
            patch("collection.piper_worker.print") as printed,
        ):
            api = api_type.return_value
            api.is_ok.return_value = True
            api.get_fps.return_value = 200.0
            api.get_state.return_value = {"teach_status": "START_RECORDING", "arm_status": "NORMAL", "err_code": 0}
            api.robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
                msg=SimpleNamespace(min_angle_limit=-1.0, max_angle_limit=1.0)
            )

            def eef_once():
                shutdown.set()
                return SimpleNamespace(pose=[0.1] * 6, timestamp=1.0, hz=200.0)

            api.get_eef_state.side_effect = eef_once
            api.get_joint_state.return_value = SimpleNamespace(joints=[0.2] * 4 + [1.01, 0.2])
            run_piper_worker(config, shared, recorder, init_done=threading.Event())

        self.assertEqual(rows[0]["joint_5"], "1.010000000")
        self.assertTrue(any("J5 above maximum" in call.args[0] for call in printed.call_args_list))


if __name__ == "__main__":
    unittest.main()
