#!/usr/bin/env python3
"""Stream Vicon Position samples as JSON lines on stdout.

rectangle_repeatability.py runs in the Piper venv (Python 3.12), which cannot
import Foxy's rclpy (Python 3.8), so it launches this relay as a subprocess::

    source ros2/mocap_env.sh
    .venv-mocap/bin/python scripts/calib/mocap_stream.py --mocap-topic /vicon/Yori/Yori

Each line is {"monotonic_ns", "frame_number", "x", "y", "z", "qx", "qy", "qz", "qw"}
in receiver units (mm). monotonic_ns is the host CLOCK_MONOTONIC receipt time,
comparable with time.monotonic_ns() in other processes on this machine. Vicon
reports a lost object as an all-zero pose; it is forwarded unchanged.
Diagnostics go to stderr so stdout carries only samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ros2/src/trimorph_mocap"))
from trimorph_mocap.mocap_worker import MocapSubscriber  # noqa: E402

POSITION_TYPE = "vicon_receiver/msg/Position"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mocap-topic", dest="mocap_topics", action="append",
                        help="Topic to relay (repeatable); default /vicon/Yori/Yori, or every "
                             "supported topic with --raw-rows")
    parser.add_argument("--raw-rows", action="store_true",
                        help="Relay complete mocap CSV rows (poses and markers) for the "
                             "collection/replay recorders instead of the compact pose lines")
    args = parser.parse_args()
    topics = args.mocap_topics or ([] if args.raw_rows else ["/vicon/Yori/Yori"])

    out = sys.stdout
    sys.stdout = sys.stderr  # MocapSubscriber prints status lines.

    def on_row(row):
        if args.raw_rows:
            out.write(json.dumps(row) + "\n")
            out.flush()
            return
        if row["message_type"] != POSITION_TYPE:
            return
        out.write(json.dumps({
            "monotonic_ns": row["monotonic_ns"], "frame_number": row["frame_number"],
            "x": row["x_trans"], "y": row["y_trans"], "z": row["z_trans"],
            "qx": row["x_rot"], "qy": row["y_rot"], "qz": row["z_rot"], "qw": row["w"],
        }) + "\n")
        out.flush()

    subscriber = MocapSubscriber(on_row, topics)
    try:
        while True:
            subscriber.poll()
    except (KeyboardInterrupt, BrokenPipeError):
        return 0
    finally:
        subscriber.close()


if __name__ == "__main__":
    raise SystemExit(main())
