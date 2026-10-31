import csv
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

from data_collection.collection import mocap_worker as mocap
from data_collection.collection.config import HandOnlyCollectionConfig
from data_collection.collection.episode import EpisodeBuffer, episode_paths
from data_collection.collect_hand_only import parse_args as collect_args, worker_specs
from data_collection.replay_episode import parse_args as replay_args


POSE_TYPE = "vicon_receiver/msg/Position"
MARKER_TYPE = "vicon_receiver/msg/MarkerPositions"


def pose():
    return NS(
        header=NS(stamp=NS(sec=123, nanosec=456), frame_id="world"),
        frame_number=42, x_trans=1.25, y_trans=-2.5, z_trans=3.75,
        x_rot=0.0, y_rot=0.0, z_rot=0.6, w=0.8,
        subject_name="hand", segment_name="wrist", translation_type="mm",
    )


class MocapTests(unittest.TestCase):
    def test_raw_pose_preserves_values_and_both_timestamp_clocks(self):
        with patch.object(mocap.time, "monotonic_ns", return_value=999), patch.object(mocap.time, "time", return_value=12.5):
            row, = mocap.message_rows(pose(), "/vicon/hand/wrist", POSE_TYPE)
        self.assertEqual(row["monotonic_ns"], 999)
        self.assertEqual(row["wall_time_s"], "12.500000000")
        self.assertEqual((row["ros_stamp_sec"], row["ros_stamp_nanosec"]), (123, 456))
        self.assertEqual(row["frame_number"], 42)
        for field in ("x_trans", "y_trans", "z_trans", "x_rot", "y_rot", "z_rot", "w", "translation_type"):
            self.assertEqual(row[field], getattr(pose(), field))
        self.assertEqual(row["occluded"], "")

    def test_marker_array_has_one_row_per_marker_and_no_orientation(self):
        message = pose()
        message.markers = [NS(x_trans=1., y_trans=2., z_trans=3., marker_name=name, subject_name="hand", segment_name="wrist", occluded=occluded)
                           for name, occluded in (("a", False), ("b", True))]
        rows = list(mocap.message_rows(message, "/vicon/markers", MARKER_TYPE))
        self.assertEqual([r["marker_name"] for r in rows], ["a", "b"])
        self.assertEqual([r["occluded"] for r in rows], [False, True])
        self.assertEqual(rows[0]["monotonic_ns"], rows[1]["monotonic_ns"])
        for row in rows:
            for field in ("x_rot", "y_rot", "z_rot", "w", "translation_type"):
                self.assertEqual(row[field], "")
        message.markers = []
        self.assertEqual(list(mocap.message_rows(message, "/markers", MARKER_TYPE)), [])

    def test_discovery_filters_types_and_explicit_topics_and_avoids_duplicates(self):
        subscriber = mocap.MocapSubscriber.__new__(mocap.MocapSubscriber)
        subscriber.node = MagicMock()
        subscriber.node.get_topic_names_and_types.return_value = [
            ("/camera", ["sensor_msgs/msg/Image"]),
            ("/hand", [POSE_TYPE]), ("/markers", [MARKER_TYPE]),
        ]
        subscriber.subscriptions = {}
        subscriber.requested = {"/hand"}
        subscriber.types = {"Position": object(), "MarkerPositions": object()}
        subscriber.qos = object()
        subscriber.on_row = MagicMock()
        subscriber.error = None
        subscriber.discover()
        subscriber.discover()
        self.assertEqual(set(subscriber.subscriptions), {"/hand"})
        self.assertTrue(subscriber.ready())
        subscriber.node.create_subscription.assert_called_once()
        callback = subscriber.node.create_subscription.call_args.args[2]
        callback(pose())
        self.assertEqual(subscriber.on_row.call_args.args[0]["topic"], "/hand")
        subscriber.requested = set()
        subscriber.discover()
        self.assertEqual(set(subscriber.subscriptions), {"/hand", "/markers"})

    def test_worker_records_only_active_episodes_and_resets_indices(self):
        shutdown = threading.Event()
        session_a = NS(elapsed_s=lambda ns: 0.25, append_mocap=MagicMock())
        session_b = NS(elapsed_s=lambda ns: 0.5, append_mocap=MagicMock())
        sessions = iter([None, session_a, session_a, None, session_b])
        recorder = NS(current=lambda: next(sessions))
        with patch.object(mocap, "MocapSubscriber") as factory:
            subscriber = factory.return_value
            count = 0
            def poll():
                nonlocal count
                callback = factory.call_args.args[0]
                callback(next(mocap.message_rows(pose(), "/hand", POSE_TYPE)))
                count += 1
                if count == 5:
                    shutdown.set()
            subscriber.poll.side_effect = poll
            subscriber.ready.return_value = True
            ready = threading.Event()
            mocap.run_mocap_worker(NS(mocap_topics=()), NS(shutdown=shutdown), recorder, init_done=ready)
            self.assertTrue(ready.is_set())
            subscriber.close.assert_called_once()
        self.assertEqual([call.args[0]["sample_idx"] for call in session_a.append_mocap.call_args_list], [0, 1])
        self.assertEqual(session_b.append_mocap.call_args.args[0]["sample_idx"], 0)

    def test_worker_timeout_closes_subscription(self):
        with patch.object(mocap, "MocapSubscriber") as factory, patch.object(mocap.time, "monotonic", side_effect=[0., 11.]):
            factory.return_value.ready.return_value = False
            with self.assertRaises(TimeoutError):
                mocap.run_mocap_worker(NS(mocap_topics=()), NS(shutdown=threading.Event()), NS(), init_done=threading.Event())
            factory.return_value.close.assert_called_once()

    def test_replay_filters_interval_and_saves_csv_on_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = mocap.ReplayMocapRecorder(Path(tmp) / "nested" / "mocap.csv")
            recorder.on_row({"monotonic_ns": 0})  # before recording
            recorder.start(10., 1.)
            for ns in (9_999_999_999, 10_000_000_000, 10_500_000_000, 11_000_000_001):
                row = next(mocap.message_rows(pose(), "/hand", POSE_TYPE))
                row["monotonic_ns"] = ns
                recorder.on_row(row)
            recorder.stop()  # also used on Ctrl-C and exceptions
            with recorder.output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual([r["elapsed_s"] for r in rows], ["0.000000000", "0.500000000"])
            self.assertEqual([r["sample_idx"] for r in rows], ["0", "1"])

    def test_empty_csv_has_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            mocap.write_mocap_csv(path, [])
            with path.open() as stream:
                self.assertEqual(next(csv.reader(stream)), list(mocap.MOCAP_FIELDS))

    def test_collection_is_opt_in_and_flushes_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = HandOnlyCollectionConfig(dataset_root=Path(tmp), hand_config=Path("unused.yaml"), hand_port="unused", hand_measure_rate_hz=50., warmup_time=0., episode_time=1.)
            self.assertNotIn("mocap", [name for name, _ in worker_specs(config)])
            config = replace(config, use_mocap=True)
            self.assertIn("mocap", [name for name, _ in worker_specs(config)])
            session = EpisodeBuffer(episode_paths(Path(tmp), 0), config)
            row = next(mocap.message_rows(pose(), "/hand", POSE_TYPE))
            session.append_mocap(row)
            session.mark_stopped("test")
            session.append_mocap(row)  # late callback is ignored
            with patch("data_collection.collection.episode._hand_metadata", return_value={}):
                session.flush()
            self.assertEqual(len(session.mocap_rows), 1)
            import json
            metadata = json.loads(session.paths.metadata_path.read_text())
            self.assertEqual(metadata["counts"]["mocap_rows"], 1)
            self.assertEqual(metadata["paths"]["mocap_csv"], "mocap.csv")
            self.assertEqual(metadata["mocap"]["recorded_topics"], ["/hand"])

    def test_cli_options(self):
        with patch.object(sys, "argv", ["collect", "--use-mocap", "--mocap-topic", "/hand"]):
            args = collect_args()
            self.assertTrue(args.use_mocap)
            self.assertEqual(args.mocap_topics, ["/hand"])
        with patch.object(sys, "argv", ["replay", "--use-mocap", "--mocap-output", "out.csv"]):
            args = replay_args()
            self.assertTrue(args.use_mocap)
            self.assertEqual(args.mocap_output, Path("out.csv"))
        with patch.object(sys, "argv", ["replay"]):
            self.assertFalse(replay_args().use_mocap)


if __name__ == "__main__":
    unittest.main()
