import sys
import unittest
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.calib import mocap_detection as detection


def observation(**changes):
    row = dict(
        topic="/vicon/hand/wrist", message_type=detection.POSE_TYPE,
        subject_name="hand", segment_name="wrist", marker_name="",
        x_trans=10., y_trans=20., z_trans=30.,
        x_rot=0., y_rot=0., z_rot=0., w=1.,
        occluded="", monotonic_ns=1_000_000_000,
        frame_id="world", translation_type="mm",
    )
    row.update(changes)
    return row


class MocapDetectionTests(unittest.TestCase):
    def test_orientation_rotates_local_axes_using_xyzw_order(self):
        row = observation(z_rot=2., w=2.)  # unnormalized 90-degree Z rotation
        np.testing.assert_allclose(
            detection.orientation_axes(row),
            [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], atol=1e-12,
        )

    def test_missing_invalid_and_marker_orientations_are_not_invented(self):
        for row in (observation(w=0.), observation(w=float("nan")),
                    observation(message_type="vicon_receiver/msg/MarkerPositions", w="", x_rot="", y_rot="", z_rot="")):
            self.assertIsNone(detection.orientation_axes(row))

    def test_latest_detections_hide_occluded_nonfinite_and_stale_objects(self):
        objects = detection.DetectedObjects()
        objects.update(observation())
        objects.update(observation(x_trans=11.))
        with patch.object(detection.time, "monotonic_ns", return_value=2_000_000_000):
            rows = objects.snapshot(2.)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["x_trans"], 11.)
        objects.update(observation(occluded=True))
        self.assertFalse(objects.objects)
        objects.update(observation(x_trans=float("inf")))
        self.assertFalse(objects.objects)
        objects.update(observation())
        with patch.object(detection.time, "monotonic_ns", return_value=4_000_000_000):
            self.assertEqual(objects.snapshot(2.), [])

    def test_objects_with_same_name_on_different_topics_remain_distinct(self):
        objects = detection.DetectedObjects()
        objects.update(observation())
        objects.update(observation(topic="/another/hand"))
        self.assertEqual(len(objects.objects), 2)

    def test_draw_pose_marker_and_invalid_orientation_and_clear_stale_artists(self):
        figure = plt.figure()
        try:
            view = detection.MocapView(figure, axis_length=5.)
            view.draw([
                observation(),
                observation(segment_name="other", w=0.),
                observation(message_type="vicon_receiver/msg/MarkerPositions", marker_name="tip", translation_type="", x_trans=40.),
            ])
            figure.canvas.draw()
            # Three object points and only one valid triad of orientation arrows.
            self.assertEqual(len(view.axes.collections), 6)
            self.assertIn("2 segment poses; 1 markers", view.status.get_text())
            self.assertIn("1 unavailable orientations", view.status.get_text())
            view.draw([])
            self.assertEqual(len(view.axes.collections), 0)
            self.assertEqual(len(view.axes.texts), 0)
            figure.canvas.draw()
        finally:
            plt.close(figure)

    def test_cli_topic_and_positive_display_options(self):
        with patch.object(sys, "argv", ["mocap_detection", "--topic", "/hand", "--mocap-topic", "/other", "--axis-length", "50"]):
            args = detection.parse_args()
        self.assertEqual(args.topics, ["/hand", "/other"])
        self.assertEqual(args.axis_length, 50.)
        with patch.object(sys, "argv", ["mocap_detection", "--stale-after-s", "0"]), patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                detection.parse_args()

    def test_close_window_stops_ros_receiver_and_closes_context(self):
        subscriber = MagicMock()
        with patch.object(sys, "argv", ["mocap_detection"]), patch.object(detection, "MocapSubscriber", return_value=subscriber), patch.object(plt, "show", side_effect=lambda: plt.gcf().canvas.draw()), patch("threading.Thread") as thread:
            self.assertEqual(detection.main(), 0)
            thread.return_value.start.assert_called_once()
            thread.return_value.join.assert_called_once()
            subscriber.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
