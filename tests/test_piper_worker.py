import unittest

from collection.piper_worker import piper_workspace_warning


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


if __name__ == "__main__":
    unittest.main()
