#!/usr/bin/env python3
"""Passively record synchronized Quest and Piper poses.

Example (run alongside the existing Quest teleoperation controller)::

    python scripts/calib/coordinate_record.py --output output/coords.csv --duration 30

Quest must stream the collection worker's JSON packets to --quest-port. If the
teleoperation process already owns that UDP port, forward/duplicate its packets
to a different port for this logger. This script sends no arm commands.

Positions retain the workers' native units and axes. Linear interpolation uses
host monotonic timestamps, as in replay's sample_stream; episode.py itself stores
raw samples. Receipt timestamps do not compensate for network or actuation lag.
Positions use linear interpolation; orientations use quaternion SLERP. Piper
roll/pitch/yaw are interpreted as radians with R = Rz(yaw) Ry(pitch) Rx(roll),
matching the dataset postprocessor. CSV quaternions use xyzw order.
Raw Quest/Piper position steps above --jump-threshold print warnings with
elapsed time, distance, XYZ delta, packet interval, and per-device counts.
Samples are retained unchanged; the threshold uses each device's native units.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection.collection import config as defaults
from data_collection.collection.episode import RecordingManager
from data_collection.collection.piper_worker import run_piper_worker
from data_collection.collection.quest_worker import run_quest_worker
from data_collection.collection.shared_state import SharedState
from data_collection.collection.utils import worker_wrapper


class CoordinateSession:
    """Minimal worker-compatible buffer; no hand/camera dependencies."""

    def __init__(self, jump_threshold=0.05):
        if not math.isfinite(jump_threshold) or jump_threshold <= 0:
            raise ValueError("jump_threshold must be finite and positive")
        self.jump_threshold = jump_threshold
        self.jump_counts = {"quest": 0, "piper": 0}
        self.start_ns = time.monotonic_ns()
        self.lock = threading.Lock()
        self.piper_rows: list[dict[str, object]] = []
        self.quest_rows: list[dict[str, object]] = []

    def elapsed_s(self, sample_ns):
        return (sample_ns - self.start_ns) / 1e9

    def _append_pose(self, device, rows, fields, row):
        warning = None
        with self.lock:
            if rows:
                previous = rows[-1]
                delta = tuple(float(row[field]) - float(previous[field]) for field in fields)
                distance = math.hypot(*delta)
                if distance > self.jump_threshold:
                    self.jump_counts[device] += 1
                    elapsed = self.elapsed_s(int(row["monotonic_ns"]))
                    interval = (int(row["monotonic_ns"]) - int(previous["monotonic_ns"])) / 1e9
                    warning = (
                        f"WARNING: [{device}] POSITION JUMP #{self.jump_counts[device]} "
                        f"t={elapsed:.3f}s step={distance:.4f} units dt={interval:.4f}s "
                        f"delta=({delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}) "
                        f"threshold={self.jump_threshold:g}"
                    )
            rows.append(row)
        if warning is not None:
            print(warning, flush=True)

    def append_piper(self, row):
        self._append_pose("piper", self.piper_rows, ("x", "y", "z"), row)

    def append_quest(self, row):
        self._append_pose("quest", self.quest_rows, ("px", "py", "pz"), row)

    def mark_stopped(self, reason):
        self.stop_reason = reason


def _stream(rows, fields, origin_ns):
    if len(rows) < 2:
        raise ValueError("Each device needs at least two samples")
    # Keep the last sample if timestamps coincide, preserving integer precision.
    unique = {int(row["monotonic_ns"]): row for row in rows}
    stamps = sorted(unique)
    times = np.array([(stamp - origin_ns) / 1e9 for stamp in stamps])
    values = np.array([[float(unique[stamp][key]) for key in fields] for stamp in stamps])
    if len(times) < 2 or not np.isfinite(values).all():
        raise ValueError("Stream needs distinct timestamps and finite XYZ values")
    return times, values


def _covered(times, query, max_gap_s):
    right = np.clip(np.searchsorted(times, query, side="left"), 1, len(times) - 1)
    exact = (query == times[right]) | (query == times[right - 1])
    return exact | ((times[right] - times[right - 1]) <= max_gap_s)


def synchronize_positions(piper_rows, quest_rows, *, origin_ns, rate_hz=50.0, max_gap_s=0.1, include_orientation=False):
    """Align XYZ and optionally orientations over overlap, without extrapolation."""
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be finite and positive")
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive")
    pfields = ("x", "y", "z") + (("roll", "pitch", "yaw") if include_orientation else ())
    qfields = ("px", "py", "pz") + (("qx", "qy", "qz", "qw") if include_orientation else ())
    pt, pv = _stream(piper_rows, pfields, origin_ns)
    qt, qv = _stream(quest_rows, qfields, origin_ns)
    start, end = max(pt[0], qt[0]), min(pt[-1], qt[-1])
    if end <= start:
        raise ValueError("Piper and Quest streams have no time overlap")
    query = start + np.arange(int(np.floor((end - start) * rate_hz)) + 1) / rate_hz
    query = query[query <= end]
    query = query[_covered(pt, query, max_gap_s) & _covered(qt, query, max_gap_s)]
    if len(query) < 2:
        raise ValueError("Not enough synchronized samples; check stream gaps/rates")
    # Same per-column np.interp technique as replay_episode.sample_stream.
    piper = np.column_stack([np.interp(query, pt, pv[:, i]) for i in range(3)])
    quest = np.column_stack([np.interp(query, qt, qv[:, i]) for i in range(3)])
    if include_orientation:
        piper_quats = Slerp(pt, Rotation.from_euler("xyz", pv[:, 3:]))(query).as_quat()
        quest_quats = Slerp(qt, Rotation.from_quat(qv[:, 3:]))(query).as_quat()
    rows = [
        dict(
            sample_idx=i,
            monotonic_ns=origin_ns + round(float(t) * 1e9),
            elapsed_s=f"{t:.9f}",
            **{f"piper_{axis}": float(p[j]) for j, axis in enumerate("xyz")},
            **{f"quest_{axis}": float(q[j]) for j, axis in enumerate("xyz")},
        )
        for i, (t, p, q) in enumerate(zip(query, piper, quest))
    ]
    if include_orientation:
        for row, pq, qq in zip(rows, piper_quats, quest_quats):
            for prefix, quat in (("piper", pq), ("quest", qq)):
                row.update({f"{prefix}_{field}": float(value)
                            for field, value in zip(("qx", "qy", "qz", "qw"), quat)})
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV (use -f to overwrite)")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite an existing output file")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds after workers initialize; 0 = until Ctrl-C")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Synchronized CSV sample rate")
    parser.add_argument("--max-gap-s", type=float, default=0.1, help="Largest interpolation bracket allowed")
    parser.add_argument("--jump-threshold", type=float, default=0.05,
                        help="Warn on raw position steps above this distance in each device's native units (default: 0.05)")
    parser.add_argument("--quest-host", default=defaults.DEFAULT_QUEST_HOST)
    parser.add_argument("--quest-port", type=int, default=defaults.DEFAULT_QUEST_PORT)
    parser.add_argument("--quest-startup-timeout-s", type=float, default=defaults.DEFAULT_QUEST_STARTUP_TIMEOUT_S)
    parser.add_argument("--piper-rate-hz", type=float, default=defaults.DEFAULT_PIPER_RATE_HZ)
    parser.add_argument("--piper-channel", default=defaults.DEFAULT_PIPER_CHANNEL)
    parser.add_argument("--piper-interface", default=defaults.DEFAULT_PIPER_INTERFACE)
    parser.add_argument("--piper-firmware", default=defaults.DEFAULT_PIPER_FIRMWARE)
    parser.add_argument("--piper-installation-pos", default=defaults.DEFAULT_PIPER_INSTALLATION_POS)
    parser.add_argument("--piper-payload", default=defaults.DEFAULT_PIPER_PAYLOAD)
    parser.add_argument("--teacher-mode-poll-hz", type=float, default=defaults.DEFAULT_TEACHER_MODE_POLL_HZ)
    return parser


def record(args):
    for name in ("rate_hz", "max_gap_s", "jump_threshold", "quest_startup_timeout_s", "piper_rate_hz", "teacher_mode_poll_hz"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.duration) or args.duration < 0:
        raise ValueError("duration must be finite and nonnegative")
    args.output = args.output.expanduser()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}; use -f to overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shared, recorder, errors = SharedState(), RecordingManager(), queue.Queue()
    threads = []
    ready_events = []
    session = None
    try:
        for name, worker in (("piper", run_piper_worker), ("quest", run_quest_worker)):
            ready = threading.Event()
            thread = threading.Thread(
                target=worker_wrapper,
                args=(name, errors, shared, ready, worker, args, shared, recorder),
                name=f"coordinate-{name}", daemon=True,
            )
            threads.append(thread)
            ready_events.append(ready)
            thread.start()
        deadline = time.monotonic() + max(defaults.WORKER_INIT_TIMEOUT_S, args.quest_startup_timeout_s)
        while not all(event.is_set() for event in ready_events):
            if shared.shutdown.wait(0.05):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out initializing coordinate workers")
        if not shared.shutdown.is_set():
            session = CoordinateSession(jump_threshold=args.jump_threshold)
            recorder.start(session)
            print("Recording poses; translate in 3D and rotate about multiple axes. Ctrl-C saves early.", flush=True)
            deadline = time.monotonic() + args.duration if args.duration else float("inf")
            while time.monotonic() < deadline and not shared.shutdown.wait(0.05):
                pass
    except KeyboardInterrupt:
        print("Stopping and saving available samples.")
    finally:
        shared.shutdown.set()
        for thread in threads:
            thread.join(timeout=5.0)
        recorder.stop("stopped")
    failures = []
    while not errors.empty():
        error = errors.get_nowait()
        failures.append(f"{error.name}: {error.message}")
    failures.extend(f"{thread.name}: did not shut down" for thread in threads if thread.is_alive())
    if session is not None:
        with session.lock:
            piper_rows, quest_rows = list(session.piper_rows), list(session.quest_rows)
            jump_counts = session.jump_counts.copy()
        print(f"Position jump totals (raw samples, threshold > {args.jump_threshold:g} units): "
              f"Quest={jump_counts['quest']}, Piper={jump_counts['piper']}", flush=True)
        try:
            rows = synchronize_positions(piper_rows, quest_rows, origin_ns=session.start_ns,
                                         rate_hz=args.rate_hz, max_gap_s=args.max_gap_s,
                                         include_orientation=True)
        except ValueError as exc:
            failures.append(str(exc))
        else:
            with args.output.open("w" if args.force else "x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved {len(rows)} synchronized samples to {args.output} "
                  f"(raw: Piper={len(piper_rows)}, Quest={len(quest_rows)}).")
    if failures:
        raise RuntimeError("; ".join(failures))


def main():
    parser = build_parser()
    try:
        record(parser.parse_args())
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
