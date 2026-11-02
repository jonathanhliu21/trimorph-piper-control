import time
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

from data_collection.collection.depthai_task_camera_worker import flip_frame
from data_collection.collection.depthai_utils import (
    apply_camera_control,
    frame_capture_monotonic_ns,
    host_clock_offset_ns,
    timedelta_ns,
)


class DepthAICameraControlTests(unittest.TestCase):
    def test_180_flip_uses_both_axes(self):
        cv2 = Mock()
        frame = object()
        flipped = object()
        cv2.flip.return_value = flipped

        self.assertIs(flip_frame(cv2, frame, "180"), flipped)
        cv2.flip.assert_called_once_with(frame, -1)

    def test_locks_exposure_and_white_balance(self):
        camera = SimpleNamespace(initialControl=Mock())

        apply_camera_control(camera, exposure_us=5829, iso=100, white_balance_k=2634)

        camera.initialControl.setManualExposure.assert_called_once_with(5829, 100)
        camera.initialControl.setManualWhiteBalance.assert_called_once_with(2634)

    def test_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "exposure"):
            apply_camera_control(None, exposure_us=0, iso=100, white_balance_k=2634)
        with self.assertRaisesRegex(ValueError, "ISO"):
            apply_camera_control(None, exposure_us=5829, iso=50, white_balance_k=2634)
        with self.assertRaisesRegex(ValueError, "white balance"):
            apply_camera_control(None, exposure_us=5829, iso=100, white_balance_k=500)

    def test_timedelta_ns_is_exact(self):
        self.assertEqual(
            timedelta_ns(timedelta(days=2, seconds=3, microseconds=456789)),
            (2 * 86_400 + 3) * 1_000_000_000 + 456_789_000,
        )

    def test_frame_time_is_mid_exposure_on_the_host_clock(self):
        middle = object()
        dai = SimpleNamespace(CameraExposureOffset=SimpleNamespace(MIDDLE=middle))
        frame = Mock()
        frame.getTimestamp.return_value = timedelta(seconds=10, microseconds=5)

        self.assertEqual(
            frame_capture_monotonic_ns(dai, frame, 7), 10_000_005_000 + 7
        )
        frame.getTimestamp.assert_called_once_with(middle)

    def test_host_clock_offset_maps_depthai_clock_to_monotonic_ns(self):
        lag_ns = 3_000_000_000
        dai = SimpleNamespace(Clock=SimpleNamespace(
            now=lambda: timedelta(microseconds=(time.monotonic_ns() - lag_ns) // 1000)
        ))

        self.assertAlmostEqual(host_clock_offset_ns(dai), lag_ns, delta=1_000_000)


if __name__ == "__main__":
    unittest.main()
