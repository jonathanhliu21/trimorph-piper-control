import unittest
from types import SimpleNamespace

import numpy as np
from midas_hand_api.actuators import control_table as ct

from collection.leader_hand_worker import (
    FORCE_FEEDBACK_FULL_SCALE_ERROR_RAD,
    FORCE_FEEDBACK_MAX_CURRENT_MA,
    TrimorphForceFeedbackController,
    force_feedback_current_ma,
)


class FakeClient:
    def __init__(self):
        self.sync_writes = []
        self.position_writes = []

    def sync_write(self, motor_ids, values, address, size):
        self.sync_writes.append((list(motor_ids), list(values), address, size))

    def write_desired_pos(self, motor_ids, positions):
        self.position_writes.append((list(motor_ids), list(positions)))


class FakeHand:
    def __init__(self):
        self.motor_ids = [4, 5, 6]
        self.dxl_client = FakeClient()
        self.config = SimpleNamespace(
            joint_signs_array=np.asarray([1.0, -1.0, 1.0]),
            home_offsets_array=np.asarray([0.1, 0.2, 0.3]),
            current_unit_ma=1.0,
            max_current_limit_ma=910.0,
        )
        self.enabled = []
        self.disabled = []

    def enable_torque(self, motor_ids):
        self.enabled.append(list(motor_ids))

    def disable_torque(self, motor_ids):
        self.disabled.append(list(motor_ids))


class ForceFeedbackCurrentTests(unittest.TestCase):
    def test_linear_opposing_current_and_saturation(self):
        self.assertEqual(force_feedback_current_ma(0.0), 0.0)
        half_scale_error = FORCE_FEEDBACK_FULL_SCALE_ERROR_RAD / 2.0
        half_current = FORCE_FEEDBACK_MAX_CURRENT_MA / 2.0
        self.assertEqual(
            force_feedback_current_ma(half_scale_error), -half_current
        )
        self.assertEqual(
            force_feedback_current_ma(-half_scale_error), half_current
        )
        self.assertEqual(
            force_feedback_current_ma(1.0), -FORCE_FEEDBACK_MAX_CURRENT_MA
        )

    def test_engages_only_contact_joint_and_applies_joint_sign(self):
        hand = FakeHand()
        controller = TrimorphForceFeedbackController(hand)

        controller.update({4: 0.03, 5: 0.0}, [0.4, 0.5, 0.6])

        self.assertEqual(hand.disabled, [[4]])
        self.assertEqual(hand.enabled, [[4]])
        self.assertIn(
            ([4], [ct.OPERATING_MODE_CURRENT], ct.ADDR_OPERATING_MODE, 1),
            hand.dxl_client.sync_writes,
        )
        self.assertEqual(
            hand.dxl_client.sync_writes[-1][0:2],
            ([4], [-FORCE_FEEDBACK_MAX_CURRENT_MA]),
        )

        controller.update({4: 0.03, 5: 0.03}, [0.4, 0.5, 0.6])
        self.assertEqual(hand.disabled, [[4], [5]])
        # Motor 5 has a negative calibration sign, so its raw current is reversed.
        self.assertEqual(
            hand.dxl_client.sync_writes[-1][0:2],
            ([5], [FORCE_FEEDBACK_MAX_CURRENT_MA]),
        )

    def test_release_restores_mode_five_and_stays_torque_free(self):
        hand = FakeHand()
        controller = TrimorphForceFeedbackController(hand)
        controller.update({4: 0.03}, [0.4, 0.5, 0.6])

        controller.update({4: 0.0}, [0.45, 0.5, 0.6])

        self.assertEqual(hand.enabled, [[4]])
        self.assertEqual(hand.disabled, [[4], [4]])
        self.assertIn(
            (
                [4],
                [ct.OPERATING_MODE_CURRENT_BASED_POSITION],
                ct.ADDR_OPERATING_MODE,
                ct.LEN_OPERATING_MODE,
            ),
            hand.dxl_client.sync_writes,
        )
        self.assertEqual(hand.dxl_client.position_writes, [([4], [0.55])])
        self.assertEqual(
            hand.dxl_client.sync_writes[-1],
            (
                [4],
                [FORCE_FEEDBACK_MAX_CURRENT_MA],
                ct.ADDR_GOAL_CURRENT,
                ct.LEN_GOAL_CURRENT,
            ),
        )

    def test_shutdown_restores_every_active_joint(self):
        hand = FakeHand()
        controller = TrimorphForceFeedbackController(hand)
        controller.update({4: 0.01, 5: -0.01}, [0.4, 0.5, 0.6])

        controller.shutdown()

        self.assertEqual(hand.enabled, [[4], [5]])
        self.assertEqual(hand.disabled, [[4], [5], [4], [5]])


if __name__ == "__main__":
    unittest.main()
