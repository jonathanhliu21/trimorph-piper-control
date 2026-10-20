import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from data_collection.collection.depthai_task_camera_worker import flip_frame
from data_collection.collection.depthai_utils import apply_camera_control


class DepthAICameraControlTests(unittest.TestCase):
    def test_180_flip_uses_both_axes(self):
        cv2 = Mock()
        frame = object()
        flipped = object()
        cv2.flip.return_value = flipped

        self.assertIs(flip_frame(cv2, frame, "180"), flipped)
        cv2.flip.assert_called_once_with(frame, -1)

    def test_applies_brightness_and_anti_banding(self):
        mode = object()
        dai = SimpleNamespace(
            CameraControl=SimpleNamespace(
                AntiBandingMode=SimpleNamespace(MAINS_60_HZ=mode)
            )
        )
        camera = SimpleNamespace(initialControl=Mock())

        apply_camera_control(dai, camera, brightness=-2, anti_banding_hz="60")

        camera.initialControl.setBrightness.assert_called_once_with(-2)
        camera.initialControl.setAntiBandingMode.assert_called_once_with(mode)

    def test_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "brightness"):
            apply_camera_control(None, None, brightness=20, anti_banding_hz="60")
        with self.assertRaisesRegex(ValueError, "anti-banding"):
            apply_camera_control(None, None, brightness=0, anti_banding_hz="bad")


if __name__ == "__main__":
    unittest.main()
