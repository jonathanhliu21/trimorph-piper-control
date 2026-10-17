import json
import socket
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from collection.quest_worker import _decode_packet, run_quest_worker


VALID_PACKET = {
    "seq": 42,
    "px": 0.1,
    "py": -0.2,
    "pz": 1.25,
    "qx": 0.0,
    "qy": 0.5,
    "qz": -0.5,
    "qw": 0.707,
}


class QuestWorkerTests(unittest.TestCase):
    def make_config(self, timeout_s=5.0):
        return SimpleNamespace(
            quest_host="0.0.0.0",
            quest_port=5005,
            quest_startup_timeout_s=timeout_s,
        )

    def test_valid_packet_is_recorded_without_pose_transforms(self):
        shutdown = threading.Event()
        shared = SimpleNamespace(shutdown=shutdown)
        rows = []
        session = SimpleNamespace(
            elapsed_s=lambda sample_ns: sample_ns / 1e9,
            append_quest=rows.append,
        )
        recorder = SimpleNamespace(current=lambda: session)
        init_done = threading.Event()
        fake_socket = MagicMock()

        def receive_once(_size):
            shutdown.set()
            return json.dumps(VALID_PACKET).encode("utf-8"), ("192.0.2.1", 1234)

        fake_socket.recvfrom.side_effect = receive_once
        with (
            patch("collection.quest_worker.socket.socket") as socket_type,
            patch("collection.quest_worker.time.monotonic_ns", return_value=2_000_000_000),
            patch("collection.quest_worker.time.time", return_value=1234.5),
        ):
            socket_type.return_value.__enter__.return_value = fake_socket
            run_quest_worker(
                self.make_config(), shared, recorder, init_done=init_done
            )

        fake_socket.bind.assert_called_once_with(("0.0.0.0", 5005))
        self.assertTrue(init_done.is_set())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_idx"], 0)
        self.assertEqual(rows[0]["monotonic_ns"], 2_000_000_000)
        self.assertEqual(rows[0]["wall_time_s"], "1234.500000")
        self.assertEqual(rows[0]["elapsed_s"], "2.000000000")
        for field, value in VALID_PACKET.items():
            self.assertEqual(rows[0][field], value)

    def test_startup_fails_when_no_packet_arrives_before_timeout(self):
        fake_socket = MagicMock()
        with (
            patch("collection.quest_worker.socket.socket") as socket_type,
            patch("collection.quest_worker.time.monotonic", side_effect=[10.0, 10.1]),
        ):
            socket_type.return_value.__enter__.return_value = fake_socket
            with self.assertRaisesRegex(TimeoutError, "No valid Quest UDP data"):
                run_quest_worker(
                    self.make_config(timeout_s=0.05),
                    SimpleNamespace(shutdown=threading.Event()),
                    SimpleNamespace(current=lambda: None),
                    init_done=threading.Event(),
                )

    def test_post_startup_socket_timeout_is_tolerated(self):
        shutdown = threading.Event()
        fake_socket = MagicMock()
        calls = 0

        def receive(_size):
            nonlocal calls
            calls += 1
            if calls == 1:
                return json.dumps(VALID_PACKET).encode("utf-8"), ("127.0.0.1", 1)
            shutdown.set()
            raise socket.timeout

        fake_socket.recvfrom.side_effect = receive
        with patch("collection.quest_worker.socket.socket") as socket_type:
            socket_type.return_value.__enter__.return_value = fake_socket
            init_done = threading.Event()
            run_quest_worker(
                self.make_config(),
                SimpleNamespace(shutdown=shutdown),
                SimpleNamespace(current=lambda: None),
                init_done=init_done,
            )

        self.assertTrue(init_done.is_set())
        self.assertEqual(calls, 2)

    def test_malformed_and_incomplete_packets_fail(self):
        with self.assertRaisesRegex(ValueError, "Invalid Quest UDP JSON"):
            _decode_packet(b"not json")

        incomplete = dict(VALID_PACKET)
        del incomplete["qw"]
        with self.assertRaisesRegex(ValueError, "missing fields qw"):
            _decode_packet(json.dumps(incomplete).encode("utf-8"))

        invalid = dict(VALID_PACKET, px=float("nan"))
        with self.assertRaisesRegex(ValueError, "px must be a finite number"):
            _decode_packet(json.dumps(invalid).encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
