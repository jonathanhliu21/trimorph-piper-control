#!/usr/bin/env python3
"""Run a Quest calibration motion sequence on Piper while logging FK, Quest and Vicon.

    python scripts/calib/quest_calibration_sequence.py --quest-port 5005 --no-mocap
    python scripts/calib/quest_calibration_sequence.py --home-only          # just move to home
    python scripts/calib/quest_calibration_sequence.py --list               # print the plan, no hardware

Then fit and validate: ``python scripts/calib/fit_quest_calibration.py <run_dir>``.

The arm always starts from the same home joint configuration (--home-joints-deg)
and is moved there in joint space before the sequence, before every segment and
at the end. Segments, in order:

  warmup          hold still at home (--warmup-s) so Quest tracking settles
  rect_<plane>    rectangle, corner at home, in each --planes plane of the Piper base
  circle_<plane>  circle through home in each plane
  rot_j<N>        sinusoidal sweep of wrist joint N (rotates the tool about a new axis)
  dynamic         smooth multi-sine motion of all six joints (translation + rotation
                  at varying speed; "dynamic trajectory" excitation)

Shapes are Cartesian move_eef paths at constant orientation; rotations and the
dynamic segment are streamed joint targets (bounded steps from measured joints,
as in replay_trajectory.py --joint-level). Joint-space segments are checked
against joint limits and the wrist singularity before the arm moves; Cartesian
segments are checked live, and a segment whose tracking error exceeds its limit
is stopped, marked failed, and the sequence continues from home.

Logged like rectangle_repeatability.py: piper.csv (joints, joint-feedback FK
fk_*, firmware pose fw_*, status), commands.csv (segment, label, mode, target),
quest.csv, mocap.csv and meta.json in --output-root/<timestamp>_<label>/.
Ctrl-C stops immediately and saves; the arm then holds its last target and is
not moved home.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection.collection import config as defaults  # noqa: E402
from scripts.calib.rectangle_repeatability import (  # noqa: E402
    PIPER_FIELDS, MOCAP_FIELDS, QUEST_FIELDS, PLANES, POSITION_MODE_SETTLE_S,
    MocapRelay, PiperLogger, QuestReceiver, git_commit, joint_fk, write_csv,
)
from scripts.piper_api.piper_api import PiperAPI  # noqa: E402
from scripts.piper_api.replay_trajectory import command_bounded_joint_step, read_joint_angles  # noqa: E402

# Last good pose from the 2026-10-31 rectangle runs (J5 well clear of 0 and its limits).
DEFAULT_HOME_DEG = (2.4, 161.4, -91.8, -8.1, 25.6, 105.0)
JOINT_FIELDS = tuple(f"joint_{joint}" for joint in range(1, 7))
COMMAND_FIELDS = ("monotonic_ns", "segment", "label", "mode",
                  *defaults.RAW_EEF_FIELDS, *(f"cmd_{name}" for name in JOINT_FIELDS))
# Two incommensurate frequencies per joint (Hz at --dynamic-speed 1); amplitudes split between them.
DYNAMIC_FREQUENCIES_HZ = ((0.11, 0.29), (0.13, 0.31), (0.17, 0.37), (0.19, 0.41), (0.23, 0.43), (0.07, 0.47))
JOINT_STEP_LIMIT_RAD = 0.05
HOME_TOLERANCE_DEG = 1.0
HOME_SETTLE_S = 1.0
SEGMENT_HOLD_S = 1.0
WRIST_SINGULARITY_MIN_DEG = 5.0


class SegmentFailed(RuntimeError):
    pass


@dataclass
class Segment:
    label: str
    mode: str  # "hold", "cart" or "joint"
    duration_s: float
    # cart: offset(t) -> xyz offset [m] from the start pose; joint: offset(t) -> 6 joint offsets [rad]
    offset: Callable[[float], np.ndarray] = field(repr=False, default=lambda t: np.zeros(3))
    max_joint_excursion_rad: np.ndarray | None = field(repr=False, default=None)


def envelope(t, duration, ramp):
    """Raised-cosine fade in/out so joint segments start and end at rest at home."""
    if ramp <= 0:
        return 1.0
    edge = min(t, duration - t)
    return 0.0 if edge <= 0 else 1.0 if edge >= ramp else 0.5 - 0.5 * math.cos(math.pi * edge / ramp)


def rectangle_offset(u, v, width, height):
    perimeter = 2 * (width + height)
    corners = [(0, 0), (width, 0), (width, height), (0, height), (0, 0)]
    lengths = [width, height, width, height]

    def at(s):
        distance = (s % 1.0) * perimeter
        for edge, ((a, b), length) in enumerate(zip(zip(corners, corners[1:]), lengths)):
            if distance <= length or edge == len(lengths) - 1:
                alpha = min(distance / length, 1.0) if length else 0.0
                point = np.zeros(3)
                point[u] = a[0] + alpha * (b[0] - a[0])
                point[v] = a[1] + alpha * (b[1] - a[1])
                return point
            distance -= length
        return np.zeros(3)
    return at, perimeter


def circle_offset(u, v, radius):
    def at(s):
        point = np.zeros(3)
        point[u] = radius - radius * math.cos(2 * math.pi * s)
        point[v] = radius * math.sin(2 * math.pi * s)
        return point
    return at, 2 * math.pi * radius


def cartesian_segment(label, path, perimeter, loops, speed):
    loop_s = perimeter / speed
    moving_s = loops * loop_s
    duration = moving_s + 2 * SEGMENT_HOLD_S

    def offset(t):
        moving = min(max(t - SEGMENT_HOLD_S, 0.0), moving_s)
        return np.zeros(3) if moving >= moving_s else path(moving / loop_s)
    return Segment(label, "cart", duration, offset)


def joint_segment(label, duration, ramp, deltas):
    """deltas(t) -> raw 6-vector before the envelope."""
    def offset(t):
        return envelope(t, duration, ramp) * deltas(t)
    grid = np.linspace(0, duration, max(200, int(duration * 50)))
    samples = np.array([offset(t) for t in grid])
    return Segment(label, "joint", duration, offset, max_joint_excursion_rad=np.array([samples.min(0), samples.max(0)]))


def build_segments(args):
    segments = []
    if args.warmup_s > 0:
        segments.append(Segment("warmup", "hold", args.warmup_s))
    size, speed = args.size_mm / 1000, args.speed_mm_s / 1000
    for plane in args.planes:
        u, v = PLANES[plane]
        if "rect" in args.shapes:
            path, perimeter = rectangle_offset(u, v, size, size)
            segments.append(cartesian_segment(f"rect_{plane}", path, perimeter, args.loops, speed))
        if "circle" in args.shapes:
            path, perimeter = circle_offset(u, v, size / 2)
            segments.append(cartesian_segment(f"circle_{plane}", path, perimeter, args.loops, speed))
    amplitude = math.radians(args.rotation_deg)
    for joint in args.rotation_joints:
        def sweep(t, index=joint - 1):
            delta = np.zeros(6)
            delta[index] = amplitude * math.sin(2 * math.pi * t / args.rotation_period_s)
            return delta
        segments.append(joint_segment(f"rot_j{joint}", args.rotation_cycles * args.rotation_period_s,
                                      min(1.0, args.rotation_period_s / 4), sweep))
    if args.dynamic_s > 0:
        amplitudes = np.radians(args.dynamic_amplitude_deg)
        phases = np.random.default_rng(args.dynamic_seed).uniform(0, 2 * math.pi, (6, 2))

        def multisine(t):
            return np.array([amplitudes[j] / 2 * sum(math.sin(2 * math.pi * args.dynamic_speed * f * t + phases[j, k])
                                                      for k, f in enumerate(DYNAMIC_FREQUENCIES_HZ[j]))
                             for j in range(6)])
        segments.append(joint_segment("dynamic", args.dynamic_s, 3.0, multisine))
    if args.segments:
        wanted = set(args.segments)
        unknown = wanted - {segment.label for segment in segments}
        if unknown:
            raise ValueError(f"Unknown --segments {sorted(unknown)}; available: {[s.label for s in segments]}")
        segments = [segment for segment in segments if segment.label in wanted]
    return segments


def check_joint_segments(home, segments, margin_deg):
    """Reject the plan before moving if any joint-space segment could leave the safe range."""
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD

    limits = ROBOT_JOINT_LIMIT_PRESET_RAD["piper"]
    margin = math.radians(margin_deg)
    problems = []
    for joint in range(6):
        low, high = limits[f"joint{joint + 1}"]
        if not low + margin <= home[joint] <= high - margin:
            problems.append(f"home J{joint + 1}={math.degrees(home[joint]):.1f} deg is within {margin_deg:g} deg of its limits")
    for segment in segments:
        if segment.max_joint_excursion_rad is None:
            continue
        lowest = home + segment.max_joint_excursion_rad[0]
        highest = home + segment.max_joint_excursion_rad[1]
        for joint in range(6):
            low, high = limits[f"joint{joint + 1}"]
            if lowest[joint] < low + margin or highest[joint] > high - margin:
                problems.append(f"{segment.label}: J{joint + 1} spans {math.degrees(lowest[joint]):.1f}.."
                                f"{math.degrees(highest[joint]):.1f} deg, limits {math.degrees(low):.0f}.."
                                f"{math.degrees(high):.0f} (margin {margin_deg:g})")
        if lowest[4] * highest[4] <= 0 or min(abs(lowest[4]), abs(highest[4])) < math.radians(WRIST_SINGULARITY_MIN_DEG):
            problems.append(f"{segment.label}: J5 spans {math.degrees(lowest[4]):.1f}..{math.degrees(highest[4]):.1f} deg, "
                            f"too close to the wrist singularity at 0")
    if problems:
        raise ValueError("Unsafe joint-space plan:\n  " + "\n  ".join(problems)
                         + "\nReduce --rotation-deg / --dynamic-amplitude-deg or change --home-joints-deg.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="quest_calib")
    parser.add_argument("--output-root", type=Path, default=Path("output/quest_calibration"))
    parser.add_argument("--home-joints-deg", type=float, nargs=6, default=DEFAULT_HOME_DEG)
    parser.add_argument("--home-speed-deg-s", type=float, default=10.0)
    parser.add_argument("--home-only", action="store_true", help="Move to home and exit.")
    parser.add_argument("--list", action="store_true", help="Print the segment plan and exit (no hardware).")
    parser.add_argument("--segments", nargs="+", help="Run only these segment labels (see --list).")
    parser.add_argument("--warmup-s", type=float, default=30.0)
    parser.add_argument("--planes", nargs="+", choices=sorted(PLANES), default=["xy", "xz", "yz"])
    parser.add_argument("--shapes", nargs="+", choices=("rect", "circle"), default=["rect", "circle"])
    parser.add_argument("--size-mm", type=float, default=30.0, help="Rectangle edge / circle diameter.")
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--speed-mm-s", type=float, default=15.0)
    parser.add_argument("--rotation-joints", type=int, nargs="*", choices=(4, 5, 6), default=[4, 5, 6])
    parser.add_argument("--rotation-deg", type=float, default=10.0)
    parser.add_argument("--rotation-period-s", type=float, default=6.0)
    parser.add_argument("--rotation-cycles", type=int, default=2)
    parser.add_argument("--dynamic-s", type=float, default=60.0, help="0 skips the dynamic segment.")
    parser.add_argument("--dynamic-amplitude-deg", type=float, nargs=6, default=(6, 5, 5, 8, 8, 8))
    parser.add_argument("--dynamic-speed", type=float, default=1.0, help="Frequency multiplier for the dynamic segment.")
    parser.add_argument("--dynamic-seed", type=int, default=0)
    parser.add_argument("--joint-margin-deg", type=float, default=3.0,
                        help="Required margin to joint limits for joint-space segments.")
    parser.add_argument("--max-tracking-error-mm", type=float, default=10.0, help="Cartesian segments.")
    parser.add_argument("--max-joint-error-deg", type=float, default=5.0, help="Joint-space segments.")
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--log-rate-hz", type=float, default=200.0)
    parser.add_argument("--speed-percent", type=int, default=100)
    parser.add_argument("--mocap-topic", default="/vicon/Yori/Yori")
    parser.add_argument("--no-mocap", action="store_true")
    parser.add_argument("--quest-port", type=int, help="Record Quest UDP packets on this port.")
    parser.add_argument("--quest-host", default=defaults.DEFAULT_QUEST_HOST)
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    parser.add_argument("--interface", default=defaults.DEFAULT_PIPER_INTERFACE)
    parser.add_argument("--channel", default=defaults.DEFAULT_PIPER_CHANNEL)
    parser.add_argument("--firmware", choices=("default", "v183", "v188"), default=defaults.DEFAULT_PIPER_FIRMWARE)
    parser.add_argument("--installation-pos", choices=("unchanged", "horizontal", "left", "right"),
                        default=defaults.DEFAULT_PIPER_INSTALLATION_POS)
    parser.add_argument("--payload", choices=("unchanged", "empty", "half", "full"), default=defaults.DEFAULT_PIPER_PAYLOAD)
    args = parser.parse_args(argv)
    positive = ("home_speed_deg_s", "size_mm", "speed_mm_s", "rotation_period_s", "dynamic_speed", "rate_hz",
                "log_rate_hz", "max_tracking_error_mm", "max_joint_error_deg")
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("warmup_s", "dynamic_s", "rotation_deg", "joint_margin_deg"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.size_mm > 60:
        parser.error("--size-mm is limited to 60 mm")
    if args.loops < 1 or args.rotation_cycles < 1:
        parser.error("--loops and --rotation-cycles must be at least 1")
    if not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be between 1 and 100")
    return args


class Sequencer:
    def __init__(self, args, api, piper, mocap, quest):
        self.args, self.api, self.piper, self.mocap, self.quest = args, api, piper, mocap, quest
        self.home = np.radians(args.home_joints_deg)
        self.period_s = 1.0 / args.rate_hz
        self.commands = []
        self.segment_index = -1

    def record(self, label, mode, pose=None, joints=None):
        row = {"monotonic_ns": time.monotonic_ns(), "segment": self.segment_index, "label": label, "mode": mode}
        if pose is not None:
            row.update({name: f"{value:.9f}" for name, value in zip(defaults.RAW_EEF_FIELDS, pose)})
        if joints is not None:
            row.update({f"cmd_{name}": f"{value:.9f}" for name, value in zip(JOINT_FIELDS, joints)})
        self.commands.append(row)

    def check_logger(self):
        if self.piper.error is not None:
            raise RuntimeError(f"Piper logging failed: {self.piper.error}") from self.piper.error

    def go_home(self):
        """Joint-space ramp from measured joints to home at --home-speed-deg-s, then verify."""
        start = np.array(read_joint_angles(self.api))
        largest = float(np.max(np.abs(self.home - start)))
        step = math.radians(self.args.home_speed_deg_s) * self.period_s
        steps = max(1, math.ceil(largest / step))
        t0 = time.monotonic()
        for tick in range(1, steps + 1 + int(HOME_SETTLE_S / self.period_s)):
            target = start + (self.home - start) * min(tick / steps, 1.0)
            command_bounded_joint_step(self.api, target, JOINT_STEP_LIMIT_RAD)
            self.record("home", "joint", joints=target)
            self.check_logger()
            sleep_s = t0 + tick * self.period_s - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
        error = np.degrees(np.max(np.abs(np.array(read_joint_angles(self.api)) - self.home)))
        if error > HOME_TOLERANCE_DEG:
            raise RuntimeError(f"Arm did not reach home (worst joint error {error:.2f} deg); "
                               f"status {self.piper.latest_status()}")

    def run_segment(self, segment):
        start_pose = np.array(self.api.get_eef_pose())
        start_fk = np.array(self.piper.latest_pose())
        t0 = time.monotonic()
        tick = 0
        while True:
            t = tick * self.period_s
            if t > segment.duration_s:
                return
            if segment.mode == "joint":
                target = self.home + segment.offset(t)
                command_bounded_joint_step(self.api, target, JOINT_STEP_LIMIT_RAD)
                self.record(segment.label, "joint", joints=target)
                error = np.degrees(np.max(np.abs(np.array(read_joint_angles(self.api)) - target)))
                if error > self.args.max_joint_error_deg:
                    raise SegmentFailed(f"joint error {error:.1f} deg > {self.args.max_joint_error_deg:g}")
            else:
                offset = segment.offset(t) if segment.mode == "cart" else np.zeros(3)
                target = np.concatenate([start_pose[:3] + offset, start_pose[3:]])
                self.api.move_eef(target.tolist())
                self.record(segment.label, segment.mode, pose=target)
                measured = self.piper.latest_pose()
                if measured is not None:
                    error = 1000 * np.linalg.norm((np.array(measured[:3]) - start_fk[:3]) - offset)
                    if error > self.args.max_tracking_error_mm:
                        raise SegmentFailed(f"tracking error {error:.1f} mm > {self.args.max_tracking_error_mm:g}; "
                                            f"status {self.piper.latest_status()}")
            self.check_logger()
            tick += 1
            sleep_s = t0 + tick * self.period_s - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)


def save(run_dir, meta, sequencer, piper, mocap, quest):
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "piper.csv", PIPER_FIELDS, piper.rows if piper else [])
    write_csv(run_dir / "commands.csv", COMMAND_FIELDS, sequencer.commands if sequencer else [])
    if mocap is not None:
        write_csv(run_dir / "mocap.csv", MOCAP_FIELDS, mocap.rows)
    if quest is not None:
        write_csv(run_dir / "quest.csv", QUEST_FIELDS, quest.rows)
    meta["counts"] = {"piper": len(piper.rows) if piper else 0,
                      "commands": len(sequencer.commands) if sequencer else 0,
                      "mocap": len(mocap.rows) if mocap else None, "quest": len(quest.rows) if quest else None}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved {run_dir} ({meta['counts']})")


def run(args):
    segments = build_segments(args)
    home = np.radians(args.home_joints_deg)
    check_joint_segments(home, segments, args.joint_margin_deg)
    total = sum(segment.duration_s for segment in segments)
    print(f"Home joints (deg): {[round(value, 1) for value in args.home_joints_deg]}")
    for segment in segments:
        print(f"  {segment.label:<12} {segment.mode:<5} {segment.duration_s:6.1f} s")
    print(f"{len(segments)} segments, {total:.0f} s of motion plus homing between segments.")
    if args.list:
        return 0

    run_dir = args.output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.label}"
    meta = {"args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "git_commit": git_commit(), "status": "started", "home_joints_rad": home.tolist(),
            "segments": [{"index": index, "label": segment.label, "mode": segment.mode,
                          "duration_s": segment.duration_s, "status": "pending"}
                         for index, segment in enumerate(segments)],
            "units": {"piper": "m, rad", "mocap": "mm (Vicon global)", "quest": "native (m)"}}
    api = mocap = quest = piper = sequencer = None
    try:
        if not args.home_only and not args.no_mocap:
            print("Starting Vicon relay...")
            mocap = MocapRelay(args.mocap_topic)
            mocap.wait_ready(20.0)
            print("Vicon object tracked.")
        if not args.home_only and args.quest_port is not None:
            quest = QuestReceiver(args.quest_host, args.quest_port)
            quest.wait_ready(defaults.DEFAULT_QUEST_STARTUP_TIMEOUT_S)
            print(f"Quest packets arriving on UDP {args.quest_port}.")

        api = PiperAPI(firmware=args.firmware, interface=args.interface, channel=args.channel,
                       installation_pos=args.installation_pos, payload=args.payload)
        api.connect()
        print(f"Piper connected: {api.is_ok()} (receive FPS: {api.get_fps():.1f})")
        if not api.is_ok():
            raise RuntimeError("Piper is not responding on the CAN bus")
        current = np.array(read_joint_angles(api))
        print(f"Current joints (deg): {np.round(np.degrees(current), 1).tolist()}; "
              f"largest move to home {np.degrees(np.max(np.abs(home - current))):.1f} deg")
        if not args.yes:
            input("Clear the workspace, keep a hand near the e-stop, and press Enter to move to home "
                  "and start (Ctrl-C aborts)...")
        if not api.enable():
            raise RuntimeError("Timed out while enabling Piper")
        api.set_position_mode()
        time.sleep(POSITION_MODE_SETTLE_S)
        api.set_speed_percent(args.speed_percent)

        piper = PiperLogger(api, args.log_rate_hz)
        piper.thread.start()
        sequencer = Sequencer(args, api, piper, mocap, quest)
        sequencer.go_home()
        print("At home.")
        if args.home_only:
            meta["status"] = "home"
            return 0
        meta["home_fk"] = piper.latest_pose()
        meta["home_pose"] = [float(value) for value in api.get_eef_pose()]
        t_start = time.monotonic()
        for index, segment in enumerate(segments):
            sequencer.segment_index = index
            info = meta["segments"][index]
            print(f"[{index + 1}/{len(segments)}] {segment.label} ({segment.duration_s:.0f} s, "
                  f"elapsed {time.monotonic() - t_start:.0f} s)", flush=True)
            sequencer.go_home()
            info["start_ns"] = time.monotonic_ns()
            try:
                sequencer.run_segment(segment)
                info["status"] = "complete"
            except SegmentFailed as exc:
                info["status"] = f"failed: {exc}"
                print(f"WARNING: {segment.label} stopped ({exc}); returning home and continuing.", flush=True)
            info["end_ns"] = time.monotonic_ns()
        sequencer.segment_index = len(segments)
        sequencer.go_home()
        meta["status"] = "complete"
        failed = [info["label"] for info in meta["segments"] if info["status"] != "complete"]
        print("Sequence complete; arm at home." + (f" Failed segments: {failed}" if failed else ""))
        return 0
    except KeyboardInterrupt:
        meta["status"] = "interrupted"
        print("\nInterrupted; the arm holds its last target (not moved home).")
        return 130
    except Exception as exc:
        meta["status"] = f"failed: {exc}"
        raise
    finally:
        if piper is not None:
            piper.stop()
        if mocap is not None:
            mocap.stop()
        if quest is not None:
            quest.stop()
        if api is not None:
            api.disconnect()
        if sequencer is not None and sequencer.commands and not args.home_only:
            save(run_dir, meta, sequencer, piper, mocap, quest)


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
