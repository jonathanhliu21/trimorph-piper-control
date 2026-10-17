import argparse
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import read_piper


class ReadPiperTests(unittest.TestCase):
    def test_once_prints_workspace_warning(self):
        class FakePiperAPI:
            def __init__(self, **kwargs):
                pass

            def connect(self):
                pass

            def disconnect(self):
                pass

            def get_joint_state(self):
                return SimpleNamespace(joints=[0.0] * 6, timestamp=1.0)

            def get_eef_state(self):
                return SimpleNamespace(pose=[0.0] * 6, timestamp=1.0)

            def get_state(self):
                return {
                    "arm_status": "TARGET_POS_EXCEEDS_LIMIT(0x4)",
                    "err_code": 0,
                }

        args = argparse.Namespace(
            firmware="v183",
            interface="socketcan",
            channel="can0",
            rate_hz=10.0,
            timeout_s=1.0,
            json=False,
            once=True,
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(read_piper, "PiperAPI", FakePiperAPI),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = read_piper.run(args)

        self.assertEqual(result, 0)
        self.assertIn("joints [rad]", stdout.getvalue())
        self.assertIn("TARGET_POS_EXCEEDS_LIMIT", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
