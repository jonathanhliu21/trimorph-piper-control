import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from data_collection.collection.quest_worker import QUEST_TRACKER_NAME
from data_collection.collection.shared_state import SharedState, TrackerSample
from data_collection.collection.tracking_monitor import (
    FAIL_COLOR,
    OK_COLOR,
    TrackingMonitor,
    display_frames,
    project,
    stream_status,
)
from data_collection.collect_hand_only import main, parse_args

SECOND = 1_000_000_000


def sample(ns, position=(0.0, 0.0, 0.0), lost=False, count=1):
    return TrackerSample(ns, tuple(position), (0.0, 0.0, 0.0, 1.0), lost, count)


class StreamStatusTests(unittest.TestCase):
    def test_green_only_for_fresh_tracked_moving_samples(self):
        now = 10 * SECOND
        moving = np.array([[0, 0, 0], [0.01, 0, 0]])
        still = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.50005]])
        self.assertEqual(stream_status(now, None, None, check_frozen=True).text, "NO DATA")
        self.assertEqual(stream_status(now, sample(now - SECOND), moving, check_frozen=True).text,
                         "NO SIGNAL 1.0s")
        self.assertEqual(stream_status(now, sample(now, lost=True), moving, check_frozen=False).text,
                         "OBJECT LOST")
        self.assertEqual(stream_status(now, sample(now), still, check_frozen=True).text, "POSITION FROZEN")
        # Vicon is not checked for stillness, and an unfilled window is not judged frozen.
        self.assertTrue(stream_status(now, sample(now), still, check_frozen=False).ok)
        self.assertTrue(stream_status(now, sample(now), None, check_frozen=True).ok)
        self.assertTrue(stream_status(now, sample(now - SECOND // 4), moving, check_frozen=True).ok)


class DisplayFrameTests(unittest.TestCase):
    def test_calibration_maps_trackers_to_piper_axes(self):
        quest, vicon = display_frames(None)
        self.assertEqual((quest.label, vicon.label), ("own axes", "own axes"))
        a = np.diag([2.0, 3.0, 4.0])
        b = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        w = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        quest, vicon = display_frames(dict(
            linear_matrix=a.tolist(), orientation=dict(rotation_matrix=b.tolist()),
            vicon_reference={"run": dict(vicon_to_piper_rotation=w.tolist())}))
        np.testing.assert_allclose(quest.linear, a)
        np.testing.assert_allclose(quest.rotation, b)
        np.testing.assert_allclose(vicon.linear, w)
        self.assertEqual((quest.label, vicon.label), ("Piper axes", "Piper axes"))

    def test_projection_puts_up_on_top_and_scales(self):
        center = (100, 100)
        origin, up, x = project([[0, 0, 0], [0, 0, 1], [1, 0, 0]], 0.0, 0.0, 50.0, center)
        np.testing.assert_array_equal(origin, center)
        np.testing.assert_array_equal(up, (100, 50))
        np.testing.assert_array_equal(x, (150, 100))


class TrackingMonitorTests(unittest.TestCase):
    def make_monitor(self, shared, **kwargs):
        import cv2

        with patch.object(cv2, "namedWindow"), patch.object(cv2, "setMouseCallback"):
            return TrackingMonitor(shared, **(dict(show_quest=True, show_vicon=True) | kwargs))

    def test_calibration_selects_only_calibrated_enabled_trackers(self):
        mapping = dict(linear_matrix=np.eye(3).tolist(),
                       orientation=dict(rotation_matrix=np.eye(3).tolist()))
        reference = {"run": dict(vicon_to_piper_rotation=np.eye(3).tolist())}
        cases = [
            (None, True, True),
            (mapping, True, False),  # Legacy Quest calibration.
            (dict(mapping, tracker="quest"), True, False),
            (dict(mapping, tracker="mocap"), False, True),
            (dict(mapping, tracker="quest", vicon_reference=reference), True, True),
            (dict(vicon_reference=reference), False, True),
            ({}, False, False),
            (dict(mapping, tracker="unknown"), False, False),
        ]
        now = 100 * SECOND
        for calibration, quest, vicon in cases:
            with self.subTest(calibration=calibration):
                shared = SharedState()
                shared.set_tracker_pose(QUEST_TRACKER_NAME, now, (0, 0, 0), (0, 0, 0, 1))
                shared.set_tracker_pose("/vicon/Yori/Yori", now, (0, 0, 0), (0, 0, 0, 1))
                monitor = self.make_monitor(shared, calibration=calibration)
                self.assertEqual((monitor.show_quest, monitor.show_vicon), (quest, vicon))
                monitor.render(now)
                expected = ({QUEST_TRACKER_NAME} if quest else set()) | (
                    {"/vicon/Yori/Yori"} if vicon else set())
                self.assertEqual(set(monitor.streams), expected)

        monitor = self.make_monitor(SharedState(), calibration=dict(
            mapping, vicon_reference=reference), show_quest=False, show_vicon=False)
        self.assertFalse(monitor.show_quest or monitor.show_vicon)

    def test_mocap_only_calibration_maps_vicon_trail_and_rotation(self):
        shared = SharedState()
        linear = np.diag([2., 3., 4.])
        rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        monitor = self.make_monitor(shared, calibration=dict(
            tracker="mocap", linear_matrix=linear.tolist(),
            orientation=dict(rotation_matrix=rotation.tolist())))
        now = 100 * SECOND
        topic = "/vicon/Yori/Yori"
        shared.set_tracker_pose(topic, now, (1, 2, 3), (0, 0, 0, 1))
        monitor.render(now)
        turn = Rotation.from_euler("x", 30, degrees=True)
        shared.set_tracker_pose(topic, now + 1, (1.1, 2.1, 3.1), tuple(turn.as_quat()))
        monitor.render(now + 1)
        stream = monitor.streams[topic]
        np.testing.assert_allclose(stream.pose[0], (0.2, 0.3, 0.4))
        np.testing.assert_allclose(stream.pose[1], rotation @ turn.as_matrix() @ rotation.T,
                                   atol=1e-12)
        self.assertEqual(stream.frame.label, "Piper axes")
        self.assertEqual(monitor.quest_frame.label, "own axes")

    def box_color(self, image, index, boxes):
        width = (image.shape[1] - 130 - 12 * (boxes + 2)) // boxes
        return tuple(int(v) for v in image[45, 12 + index * (width + 12) + width - 5])

    def test_status_boxes_turn_red_when_a_stream_drops(self):
        shared = SharedState()
        monitor = self.make_monitor(shared)
        now = 100 * SECOND
        image = monitor.render(now)
        self.assertEqual(self.box_color(image, 0, 2), FAIL_COLOR)  # Quest: no data
        self.assertEqual(self.box_color(image, 1, 2), FAIL_COLOR)  # Vicon: no topic yet

        shared.set_tracker_pose(QUEST_TRACKER_NAME, now, (0, 0, 0), (0, 0, 0, 1))
        shared.set_tracker_pose("/vicon/Yori/Yori", now, (1, 2, 3), (0, 0, 0, 1))
        image = monitor.render(now)
        self.assertEqual(self.box_color(image, 0, 2), OK_COLOR)
        self.assertEqual(self.box_color(image, 1, 2), OK_COLOR)

        shared.set_tracker_pose("/vicon/Yori/Yori", now, (0, 0, 0), (0, 0, 0, 1), lost=True)
        image = monitor.render(now + SECOND)  # Quest silent for 1 s, Vicon lost
        self.assertEqual(self.box_color(image, 0, 2), FAIL_COLOR)
        self.assertEqual(self.box_color(image, 1, 2), FAIL_COLOR)

    def test_frozen_quest_turns_red_after_one_second(self):
        shared = SharedState()
        monitor = self.make_monitor(shared)
        now = 100 * SECOND
        for step in range(25):
            t = now + step * SECOND // 20
            shared.set_tracker_pose(QUEST_TRACKER_NAME, t, (0.0934, 0.8303, 1.3493), (0, 0, 0, 1))
            image = monitor.render(t)
        self.assertEqual(self.box_color(image, 0, 2), FAIL_COLOR)

    def test_trails_are_relative_to_the_first_pose_and_mapped_by_calibration(self):
        shared = SharedState()
        a = np.diag([2.0, 2.0, 2.0])
        monitor = self.make_monitor(shared, calibration=dict(
            linear_matrix=a.tolist(), orientation=dict(rotation_matrix=np.eye(3).tolist())))
        now = 100 * SECOND
        shared.set_tracker_pose(QUEST_TRACKER_NAME, now, (1.0, 1.0, 1.0), (0, 0, 0, 1))
        monitor.render(now)
        shared.set_tracker_pose(QUEST_TRACKER_NAME, now + 1, (1.1, 1.0, 1.0), (0, 0, 0, 1))
        monitor.render(now + 1)
        stream = monitor.streams[QUEST_TRACKER_NAME]
        np.testing.assert_allclose(stream.pose[0], (0.2, 0.0, 0.0), atol=1e-12)
        self.assertEqual(len(stream.trail), 2)
        monitor._on_mouse(monitor.cv2.EVENT_LBUTTONDBLCLK, 0, 0, 0, None)
        self.assertIsNone(stream.pose)
        self.assertEqual(len(stream.trail), 0)


class MonitorCliTests(unittest.TestCase):
    def test_shared_calibration_selects_windows_and_respects_overrides(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'cal.json'
            for calibration, ik in [({'tracker': 'mocap'}, True),
                                    ({'tracker': 'quest'}, False),
                                    ({'vicon_reference': {'run': {}}}, True)]:
                path.write_text(json.dumps(calibration))
                with self.subTest(calibration=calibration), patch(
                        'sys.argv', ['collect_hand_only.py', '--calibration', str(path)]):
                    args = parse_args()
                    self.assertEqual(args.monitor_calibration, path)
                    self.assertEqual(args.mocap_ik_calibration, path if ik else None)
            with patch('sys.argv', ['collect_hand_only.py', '--calibration', str(path),
                                    '--monitor-calibration', 'view.json',
                                    '--mocap-ik-calibration', 'ik.json']):
                args = parse_args()
                self.assertEqual(args.monitor_calibration, Path('view.json'))
                self.assertEqual(args.mocap_ik_calibration, Path('ik.json'))
            with patch('sys.argv', ['collect_hand_only.py', '--calibration', str(path),
                                    '--no-monitor', '--no-mocap']):
                args = parse_args()
                self.assertIsNone(args.monitor_calibration)
                self.assertIsNone(args.mocap_ik_calibration)

    def test_monitor_follows_quest_and_mocap_and_can_be_disabled(self):
        with patch("sys.argv", ["collect_hand_only.py"]):
            self.assertTrue(parse_args().show_tracking_monitor)
        with patch("sys.argv", ["collect_hand_only.py", "--no-monitor"]):
            self.assertFalse(parse_args().show_tracking_monitor)
        with patch("sys.argv", ["collect_hand_only.py", "--no-quest", "--no-mocap"]):
            self.assertFalse(parse_args().show_tracking_monitor)
        for argv in (["--no-quest", "--no-mocap", "--monitor"],
                     ["--no-monitor", "--monitor-calibration", "cal.json"]):
            with (self.subTest(argv=argv), patch("sys.argv", ["collect_hand_only.py", *argv]),
                  patch("sys.stderr"), self.assertRaises(SystemExit)):
                parse_args()

    def test_monitor_options_reach_the_config(self):
        with (
            patch("sys.argv", ["collect_hand_only.py", "--monitor-calibration", "cal.json"]),
            patch("data_collection.collect_hand_only.run_hand_only", return_value=0) as run,
        ):
            self.assertEqual(main(), 0)
        config = run.call_args.args[0]
        self.assertTrue(config.show_tracking_monitor)
        self.assertEqual(str(config.monitor_calibration), "cal.json")


if __name__ == "__main__":
    unittest.main()
