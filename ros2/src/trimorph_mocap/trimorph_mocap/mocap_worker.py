"""Opt-in ROS 2 Vicon logging. ROS imports occur only when connecting."""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path


MOCAP_FIELDS = (
    "sample_idx", "monotonic_ns", "wall_time_s", "elapsed_s", "topic",
    "message_type", "ros_stamp_sec", "ros_stamp_nanosec", "frame_id",
    "frame_number", "subject_name", "segment_name", "marker_name",
    "x_trans", "y_trans", "z_trans", "x_rot", "y_rot", "z_rot", "w",
    "translation_type", "occluded",
)
MOCAP_TYPES = {
    "vicon_receiver/msg/Position": "Position",
    "vicon_receiver/msg/MarkerPositions": "MarkerPositions",
}


def message_rows(message, topic: str, message_type: str):
    """Flatten one source frame without changing units or inventing orientation."""
    sample_ns = time.monotonic_ns()
    wall_time_s = time.time()
    header = message.header
    base = dict.fromkeys(MOCAP_FIELDS, "")
    base.update(
        monotonic_ns=sample_ns, wall_time_s=f"{wall_time_s:.9f}",
        topic=topic, message_type=message_type,
        ros_stamp_sec=header.stamp.sec, ros_stamp_nanosec=header.stamp.nanosec,
        frame_id=header.frame_id, frame_number=message.frame_number,
    )
    is_pose = message_type == "vicon_receiver/msg/Position"
    for item in (message,) if is_pose else message.markers:
        row = base.copy()
        for field in ("subject_name", "segment_name", "x_trans", "y_trans", "z_trans"):
            row[field] = getattr(item, field)
        if is_pose:
            for field in ("x_rot", "y_rot", "z_rot", "w", "translation_type"):
                row[field] = getattr(item, field)
        else:
            row.update(marker_name=item.marker_name, occluded=item.occluded)
        yield row


def write_mocap_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MOCAP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class MocapSubscriber:
    """Own an isolated ROS context; discover only the two supported message types."""

    def __init__(self, on_row, topics=()) -> None:
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from vicon_receiver.msg import MarkerPositions, Position
        except ImportError as exc:
            raise RuntimeError(
                "Mocap requires ROS 2 rclpy and generated vicon_receiver messages; "
                "source your ROS installation and Vicon workspace before running."
            ) from exc
        self.context = Context()
        self.node = None
        self.executor = None
        self.subscriptions = {}
        self.requested = set(topics)
        self.on_row = on_row
        self.error = None
        self.types = {"Position": Position, "MarkerPositions": MarkerPositions}
        self.qos = qos_profile_sensor_data
        try:
            rclpy.init(args=[], context=self.context)
            self.node = rclpy.create_node("trimorph_mocap_recorder", context=self.context)
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
        except BaseException:
            self.close()
            raise
        self.next_discovery = 0.0

    def discover(self):
        for topic, types in self.node.get_topic_names_and_types():
            if topic in self.subscriptions or (self.requested and topic not in self.requested):
                continue
            supported = [kind for kind in types if kind in MOCAP_TYPES]
            if not supported:
                if topic in self.requested:
                    raise ValueError(f"Mocap topic {topic} has unsupported types: {types}")
                continue
            kind = supported[0]

            def callback(message, topic=topic, kind=kind):
                try:
                    for row in message_rows(message, topic, kind):
                        self.on_row(row)
                except Exception as exc:
                    self.error = exc

            self.subscriptions[topic] = self.node.create_subscription(
                self.types[MOCAP_TYPES[kind]], topic, callback, self.qos,
            )
            print(f"[mocap_worker] Subscribed to {topic} ({kind})")

    def ready(self):
        return (self.requested <= self.subscriptions.keys()) if self.requested else bool(self.subscriptions)

    def poll(self):
        if not self.context.ok():
            raise RuntimeError("Mocap ROS context shut down unexpectedly")
        if time.monotonic() >= self.next_discovery:
            self.discover()
            self.next_discovery = time.monotonic() + 0.5
        self.executor.spin_once(timeout_sec=0.1)
        if self.error is not None:
            raise RuntimeError(f"Mocap callback failed: {self.error}") from self.error

    def close(self):
        if self.executor is not None:
            self.executor.shutdown()
        if self.node is not None:
            self.node.destroy_node()
        if self.context.ok():
            self.context.shutdown()


class MocapRelay:
    """MocapSubscriber run under the Foxy venv when this Python cannot load Foxy's rclpy.

    The Piper/hand venv is Python 3.12 while Foxy's rclpy is built for 3.8, so the
    subscriber runs as scripts/calib/mocap_stream.py --raw-rows in .venv-mocap and
    sends its CSV rows as JSON lines. monotonic_ns is CLOCK_MONOTONIC, shared by
    both processes, so rows stay synchronized with the hand and camera.
    """

    def __init__(self, on_row, topics=()) -> None:
        import shlex
        import subprocess

        repo = Path(__file__).resolve().parents[4]
        topic_args = "".join(f" --mocap-topic {shlex.quote(topic)}" for topic in topics)
        # No positional args: a sourced script inherits the caller's $1 as its interface name.
        script = ("source ros2/mocap_env.sh >&2 && exec .venv-mocap/bin/python -u "
                  f"scripts/calib/mocap_stream.py --raw-rows{topic_args}")
        self.on_row = on_row
        self.rows_seen = 0
        self.error = None
        self.startup_timeout_s = 20.0  # sourcing ROS and DDS discovery take a few seconds
        self.process = subprocess.Popen(["bash", "-c", script], cwd=repo, stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, name="mocap-relay", daemon=True)
        self.thread.start()

    def _read(self):
        import json

        try:
            for line in self.process.stdout:
                self.on_row(json.loads(line))
                self.rows_seen += 1
        except Exception as exc:
            self.error = exc

    def ready(self):
        return self.rows_seen > 0

    def poll(self):
        time.sleep(0.05)
        if self.error is not None:
            raise RuntimeError(f"Mocap relay failed: {self.error}") from self.error
        if self.process.poll() is not None:
            raise RuntimeError(f"Mocap relay exited with code {self.process.returncode}; see its stderr above")

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except Exception:
                self.process.kill()


def open_subscriber(on_row, topics=()):
    """Subscribe in this process, or through MocapRelay when rclpy cannot be imported here."""
    try:
        return MocapSubscriber(on_row, topics)
    except RuntimeError as exc:
        if not isinstance(exc.__cause__, ImportError):
            raise
        print("[mocap_worker] rclpy is unavailable in this Python; relaying Vicon through .venv-mocap")
        return MocapRelay(on_row, topics)


def startup_timeout_s(subscriber) -> float:
    return subscriber.startup_timeout_s if isinstance(subscriber, MocapRelay) else 10.0


def run_mocap_worker(config, shared, recorder, *, init_done) -> None:
    sample_session = None
    sample_idx = 0

    def on_row(row):
        nonlocal sample_session, sample_idx
        session = recorder.current()
        if session is None:
            return
        if session is not sample_session:
            sample_session, sample_idx = session, 0
        row.update(sample_idx=sample_idx, elapsed_s=f"{session.elapsed_s(row['monotonic_ns']):.9f}")
        session.append_mocap(row)
        sample_idx += 1

    subscriber = open_subscriber(on_row, config.mocap_topics)
    deadline = time.monotonic() + startup_timeout_s(subscriber)
    try:
        while not shared.shutdown.is_set():
            subscriber.poll()
            if subscriber.ready():
                init_done.set()
            elif time.monotonic() >= deadline:
                raise TimeoutError("No matching mocap topics discovered within 10s (or requested topics missing)")
    finally:
        subscriber.close()


class ReplayMocapRecorder:
    """Record only the replay interval and save partial samples on interruption."""

    def __init__(self, output: Path, topics=()):
        self.output = output.expanduser().resolve()
        self.rows = []
        self.start_ns = None
        self.end_ns = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.subscriber = None
        self.topics = topics
        self.error = None

    def prepare(self):
        self.subscriber = open_subscriber(self.on_row, self.topics)
        deadline = time.monotonic() + startup_timeout_s(self.subscriber)
        while not self.subscriber.ready():
            self.subscriber.poll()
            if time.monotonic() >= deadline:
                raise TimeoutError("No matching mocap topics discovered within 10s (or requested topics missing)")
        self.thread = threading.Thread(target=self.run, name="mocap-replay", daemon=True)
        self.thread.start()

    def start(self, wall_start: float, duration_s: float):
        with self.lock:
            self.start_ns = round(wall_start * 1e9)
            self.end_ns = self.start_ns + round(duration_s * 1e9)

    def on_row(self, row):
        with self.lock:
            if self.start_ns is None or not self.start_ns <= row["monotonic_ns"] <= self.end_ns:
                return
            row.update(sample_idx=len(self.rows), elapsed_s=f"{(row['monotonic_ns'] - self.start_ns) / 1e9:.9f}")
            self.rows.append(row)

    def run(self):
        try:
            while not self.stop_event.is_set():
                self.subscriber.poll()
        except Exception as exc:
            self.error = exc

    def raise_if_failed(self):
        if self.error is not None:
            raise RuntimeError(f"Mocap recording failed: {self.error}") from self.error

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        if self.subscriber is not None:
            self.subscriber.close()
        if self.start_ns is not None:
            write_mocap_csv(self.output, self.rows)
            print(f"Saved {len(self.rows)} replay mocap rows to {self.output}")
