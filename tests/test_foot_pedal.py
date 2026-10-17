import unittest
from unittest.mock import patch

from scripts.piper_api.foot_pedal import EVIOCGRAB, FootPedalTrigger


class FootPedalTriggerTest(unittest.TestCase):
    @patch("scripts.piper_api.foot_pedal.Path.exists", return_value=True)
    @patch.object(FootPedalTrigger, "_drain_raw_events")
    @patch("scripts.piper_api.foot_pedal.fcntl.ioctl")
    @patch("scripts.piper_api.foot_pedal.os.close")
    @patch("scripts.piper_api.foot_pedal.os.open", return_value=17)
    def test_start_grabs_device_and_stop_releases_it(
        self,
        mock_open,
        mock_close,
        mock_ioctl,
        _mock_drain,
        _mock_exists,
    ):
        pedal = FootPedalTrigger("/dev/input/event-test")

        pedal.start()
        pedal.stop()

        mock_open.assert_called_once()
        self.assertEqual(
            mock_ioctl.call_args_list[0].args,
            (17, EVIOCGRAB, 1),
        )
        self.assertEqual(
            mock_ioctl.call_args_list[1].args,
            (17, EVIOCGRAB, 0),
        )
        mock_close.assert_called_once_with(17)

    @patch("scripts.piper_api.foot_pedal.Path.exists", return_value=True)
    @patch.object(FootPedalTrigger, "_drain_raw_events")
    @patch("scripts.piper_api.foot_pedal.fcntl.ioctl")
    @patch("scripts.piper_api.foot_pedal.os.close")
    @patch("scripts.piper_api.foot_pedal.os.open", return_value=18)
    def test_grab_can_be_disabled(
        self,
        _mock_open,
        mock_close,
        mock_ioctl,
        _mock_drain,
        _mock_exists,
    ):
        pedal = FootPedalTrigger(
            "/dev/input/event-test",
            suppress_keystrokes=False,
        )

        pedal.start()
        pedal.stop()

        mock_ioctl.assert_not_called()
        mock_close.assert_called_once_with(18)


if __name__ == "__main__":
    unittest.main()
