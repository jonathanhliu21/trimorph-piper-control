import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from data_collection.collection.shared_state import SharedState
from data_collection.collection.tactile_worker import TACTILE_SENSOR_NAMES, run_tactile_worker


class _Config:
    paxini_port = "/dev/paxini"
    tactile_debug_log = None


class _RecordingSharedState(SharedState):
    def __init__(self):
        super().__init__()
        self.tactile_history = []

    def set_tactile_contacts(self, contacts):
        super().set_tactile_contacts(contacts)
        self.tactile_history.append(self.latest_tactile_contacts())


class TactileWorkerTests(unittest.TestCase):
    def test_worker_detects_contacts_in_fixed_finger_order_and_disconnects(self):
        shared = _RecordingSharedState()
        init_done = unittest.mock.Mock()

        class FakePaxiniConfig:
            def __init__(self, *, port):
                self.port = port

        class FakeSensor:
            instance = None

            def __init__(self, config):
                self.config = config
                self.port = config.port
                self.connected = False
                self.disconnected = False
                FakeSensor.instance = self

            def connect(self):
                self.connected = True

            def read_tactile_fz(self):
                shared.shutdown.set()
                return {
                    name: np.asarray([index + 1.0])
                    for index, name in enumerate(TACTILE_SENSOR_NAMES)
                }

            def disconnect(self):
                self.disconnected = True

        class FakeDetector:
            def __init__(self, *, name):
                self.name = name

            def update(self, values, *, now_s):
                self.assertions = (values, now_s)
                return self.name in {"index", "ring"}

        fake_api = types.SimpleNamespace(
            PaxiniConfig=FakePaxiniConfig,
            PaxiniHandSensor=FakeSensor,
        )
        with (
            patch.dict(sys.modules, {"midas_hand_api": fake_api}),
            patch(
                "data_collection.collection.tactile_worker.TactileContactDetector", FakeDetector
            ),
        ):
            run_tactile_worker(_Config(), shared, None, init_done=init_done)

        init_done.set.assert_called_once_with()
        self.assertTrue(FakeSensor.instance.connected)
        self.assertTrue(FakeSensor.instance.disconnected)
        self.assertIn((True, False, True, False), shared.tactile_history)
        self.assertIsNone(shared.tactile_history[-1])


if __name__ == "__main__":
    unittest.main()
