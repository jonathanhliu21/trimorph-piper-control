#!/usr/bin/env python3
"""Record a Quest demonstration, then replay it on Piper in absolute coordinates.

    # No arm: record and check a demonstration (prints where the arm would go).
    python scripts/teleop/quest_replay.py --calibration CAL.json --quest-port 5005 --dry-run
    # Record (pedal press = start, press again = stop), then Enter = replay on the arm.
    python scripts/teleop/quest_replay.py --calibration CAL.json --quest-port 5005
    # Replay an earlier recording of this script.
    python scripts/teleop/quest_replay.py --calibration CAL.json --replay output/quest_replay/<session>/rec_01

CAL.json is written by scripts/calib/fit_quest_calibration.py. Unlike quest_teleop.py (deltas from a
clutch anchor), every Quest sample maps to an absolute flange target with the full calibration:

    p = A q + R l + c            A = linear_matrix, l = tracked_point_offset, c = origin of the calibration
                                 recordings (all from one Quest tracking session)
    R = R_home                   default: orientation held at home
    R = (B dRq B^T) R_home       --with-orientation: controller rotation since the first sample

so the controller-equivalent point on the flange (where the controller sat during calibration) goes
where the hand-held controller was. The origin c only holds while the Quest keeps the tracking session
of the calibration (it drifted ~1 cm over 30 min on 2026-10-31); a recentred Quest moves every target,
which the workspace check below rejects.

The arm stays at home while you record. A recording is replayed only if every target is within
--max-offset-mm / --max-rotation-deg of home, its speed (times --playback-speed) is within
--max-speed-mm-s / --max-angular-speed-deg-s, and Quest tracking has no gap over --max-gap-s or jump
over --quest-jump-mm. Before each replay the arm moves from wherever it is to the first target along a
minimum-jerk Cartesian path (--approach-speed-mm-s), holds it for --settle-s and checks it got there
within --start-tolerance-mm. The replay clock then eases in and out over --ease-s, so the arm starts and
ends at rest even if the hand was moving when the pedal was pressed. Streaming stops (the arm holds its
measured pose) on the quest_teleop.py faults: tracking error, controller limit flags, wrist singularity.

Commands (type then Enter): Enter = replay, r = record again, h = home, q = home and quit. During a
replay, the pedal or Enter stops it. Ctrl-C stops at once without moving home. Each recording is saved
to --output-root/<session>/rec_NN/ (quest.csv, trajectory.csv); arm logs go to the session directory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import signal
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection.collection import config as defaults  # noqa: E402
from scripts.calib.rectangle_repeatability import (  # noqa: E402
    PIPER_FIELDS, POSITION_MODE_SETTLE_S, QUEST_FIELDS, PiperLogger, write_csv,
)
from scripts.teleop.quest_teleop import (  # noqa: E402
    QuestStream, go_home, home_warnings, keyboard_reader, pose_list, rate_limit, saved_home_deg,
)

COMMAND_FIELDS = ("monotonic_ns", "replay", "phase", "x", "y", "z", "roll", "pitch", "yaw")
EVENT_FIELDS = ("monotonic_ns", "event", "detail")
TRAJECTORY_FIELDS = ("t", "x", "y", "z", "roll", "pitch", "yaw")
STATUS_PERIOD_S = 0.5
SPEED_WINDOW_S = 0.1  # speeds are checked over this window so Quest jitter does not dominate
MIN_JERK_PEAK = 1.875  # peak / average speed of a minimum-jerk profile


def calibration_origin(calibration):
    """Origin c of p = A q + R l + c, and the largest deviation of one recording's origin from it (m)."""
    if calibration.get("position_offset") is not None:
        return np.asarray(calibration["position_offset"], float), 0.0
    offsets = [item["position_offset"] for item in calibration.get("recordings", [])
               if item.get("position_offset") is not None]
    if not offsets:
        raise ValueError("Calibration has no position_offset; absolute mapping needs the recordings' origin")
    offsets = np.asarray(offsets, float)
    origin = offsets.mean(axis=0)
    return origin, float(np.max(np.linalg.norm(offsets - origin, axis=1)))


class AbsoluteMapper:
    """Absolute Quest pose -> Piper flange target; no I/O, so it can be tested without hardware."""

    def __init__(self, calibration, home_pose, *, with_orientation=False):
        self.a = np.asarray(calibration["linear_matrix"], float)
        self.b = np.asarray(calibration["orientation"]["rotation_matrix"], float)
        self.lever = np.asarray(calibration.get("tracked_point_offset", {}).get("vector", (0, 0, 0)), float)
        self.origin, self.origin_spread = calibration_origin(calibration)
        self.home_p = np.asarray(home_pose[:3], float)
        self.home_r = Rotation.from_euler("xyz", home_pose[3:])
        self.with_orientation = with_orientation

    def map(self, quest_positions, quest_quaternions):
        """Nx3 positions, Nx4 xyzw quaternions -> (Nx3 flange positions, N Rotations)."""
        q = np.atleast_2d(np.asarray(quest_positions, float))
        if self.with_orientation:
            rq = Rotation.from_quat(np.atleast_2d(quest_quaternions))
            rotations = Rotation.from_matrix(self.b @ (rq * rq[0].inv()).as_matrix() @ self.b.T) * self.home_r
        else:
            rotations = Rotation.from_quat(np.tile(self.home_r.as_quat(), (len(q), 1)))
        return q @ self.a.T + rotations.apply(self.lever) + self.origin, rotations


def quest_arrays(rows):
    """Quest rows (QuestStream or quest.csv) -> times from the first sample (s), positions, quaternions."""
    t = np.array([int(row["monotonic_ns"]) for row in rows]) / 1e9
    q = np.array([[float(row[name]) for name in ("px", "py", "pz")] for row in rows]).reshape(-1, 3)
    quat = np.array([[float(row[name]) for name in ("qx", "qy", "qz", "qw")] for row in rows]).reshape(-1, 4)
    norm = np.linalg.norm(quat, axis=1)
    t, q, quat = t[norm > 0.5], q[norm > 0.5], quat[norm > 0.5] / norm[norm > 0.5, None]
    t, unique = np.unique(t, return_index=True)
    return t - (t[0] if len(t) else 0.0), q[unique], quat[unique]


class Trajectory:
    """Mapped targets with interpolation (linear position, slerp rotation) in recording time."""

    def __init__(self, times, positions, rotations):
        self.times, self.positions, self.rotations = times, positions, rotations
        self.duration = float(times[-1])
        self.slerp = Slerp(times, rotations)

    def at(self, t):
        t = min(max(t, 0.0), self.duration)
        position = np.array([np.interp(t, self.times, self.positions[:, axis]) for axis in range(3)])
        return position, self.slerp([t])[0]


def check_trajectory(trajectory, quest_times, quest_positions, mapper, *, max_offset_m, max_rotation_rad,
                     max_speed_m_s, max_angular_speed_rad_s, playback_speed, max_gap_s, quest_jump_m):
    """(problems that make the replay unsafe, summary stats) for a mapped recording."""
    if len(quest_times) < 2 or trajectory is None or trajectory.duration < 2 * SPEED_WINDOW_S:
        return ["recording is shorter than 0.2 s"], {}
    offsets = np.linalg.norm(trajectory.positions - mapper.home_p, axis=1)
    rotations = (trajectory.rotations * mapper.home_r.inv()).magnitude()
    grid = np.arange(0.0, trajectory.duration, SPEED_WINDOW_S)
    samples = [trajectory.at(t) for t in grid]
    speeds = playback_speed * np.linalg.norm(np.diff([p for p, _ in samples], axis=0), axis=1) / SPEED_WINDOW_S
    angular = playback_speed * np.array([(b * a.inv()).magnitude() for (_, a), (_, b)
                                         in zip(samples, samples[1:])]) / SPEED_WINDOW_S
    gaps = np.diff(quest_times)
    jumps = np.linalg.norm(np.diff(quest_positions, axis=0), axis=1)
    stats = {"duration_s": trajectory.duration, "samples": len(quest_times),
             "start_offset_mm": 1000 * offsets[0], "max_offset_mm": 1000 * offsets.max(),
             "max_rotation_deg": math.degrees(rotations.max()), "peak_speed_mm_s": 1000 * speeds.max(),
             "peak_angular_speed_deg_s": math.degrees(angular.max()), "max_gap_s": gaps.max(),
             "max_quest_jump_mm": 1000 * jumps.max()}
    at = lambda values, times: f"at t = {times[int(np.argmax(values))]:.1f} s"
    problems = []
    if offsets.max() > max_offset_m:
        problems.append(f"target {1000 * offsets.max():.0f} mm from home {at(offsets, trajectory.times)} "
                        f"(--max-offset-mm {1000 * max_offset_m:g}); demonstrate closer to the arm's home pose")
    if rotations.max() > max_rotation_rad:
        problems.append(f"orientation {math.degrees(rotations.max()):.0f} deg from home {at(rotations, trajectory.times)} "
                        f"(--max-rotation-deg {math.degrees(max_rotation_rad):g})")
    if speeds.max() > max_speed_m_s:
        problems.append(f"peak speed {1000 * speeds.max():.0f} mm/s {at(speeds, grid)} (--max-speed-mm-s "
                        f"{1000 * max_speed_m_s:g}); lower --playback-speed")
    if angular.max() > max_angular_speed_rad_s:
        problems.append(f"peak angular speed {math.degrees(angular.max()):.0f} deg/s {at(angular, grid)} "
                        f"(--max-angular-speed-deg-s {math.degrees(max_angular_speed_rad_s):g}); lower --playback-speed")
    if gaps.max() > max_gap_s:
        problems.append(f"Quest gap of {gaps.max():.2f} s {at(gaps, quest_times)} (--max-gap-s {max_gap_s:g})")
    if jumps.max() > quest_jump_m:
        problems.append(f"Quest position jump of {1000 * jumps.max():.0f} mm {at(jumps, quest_times)} "
                        f"(--quest-jump-mm {1000 * quest_jump_m:g}); tracking was lost")
    return problems, stats


def replay_clock(elapsed, duration, playback_speed, ease_s):
    """Recording time for wall time `elapsed`: the playback rate ramps 0 -> speed -> 0 over ease_s at each end."""
    ease = min(ease_s, duration / playback_speed)
    wall = duration / playback_speed + ease
    if ease <= 0:
        return min(max(elapsed, 0.0) * playback_speed, duration)
    if elapsed <= 0:
        return 0.0
    if elapsed < ease:
        return playback_speed * elapsed ** 2 / (2 * ease)
    if elapsed < wall - ease:
        return playback_speed * (elapsed - ease / 2)
    if elapsed < wall:
        return duration - playback_speed * (wall - elapsed) ** 2 / (2 * ease)
    return duration


def replay_wall_s(duration, playback_speed, ease_s):
    return duration / playback_speed + min(ease_s, duration / playback_speed)


def approach_path(start, goal, rate_hz, max_speed_m_s, max_angular_speed_rad_s, min_duration_s=1.0):
    """Minimum-jerk poses from start to goal: zero velocity and acceleration at both ends."""
    (p_start, r_start), (p_goal, r_goal) = start, goal
    distance = np.linalg.norm(p_goal - p_start)
    angle = (r_goal * r_start.inv()).magnitude()
    duration = max(min_duration_s, MIN_JERK_PEAK * distance / max_speed_m_s,
                   MIN_JERK_PEAK * angle / max_angular_speed_rad_s)
    steps = max(1, math.ceil(duration * rate_hz))
    slerp = Slerp([0, 1], Rotation.concatenate([r_start, r_goal]))
    for step in range(1, steps + 1):
        u = step / steps
        s = u ** 3 * (10 - 15 * u + 6 * u ** 2)
        yield p_start + s * (p_goal - p_start), slerp([s])[0]


class ArmGuard:
    """The quest_teleop.py stop conditions: tracking error, controller limit flags, wrist singularity."""

    def __init__(self, args):
        self.args = args
        self.over_since = None  # time tracking error first exceeded its allowance
        self.j5_floor = None  # wrist-singularity floor: starts at |J5|, ratchets up to the margin

    def check(self, piper, target_position, speed_mm_s, now):
        measured = piper.latest_pose()
        if measured is None:
            return None
        args = self.args
        error = 1000 * np.linalg.norm(np.asarray(measured[:3]) - target_position)
        allowed = args.max_tracking_error_mm + speed_mm_s * args.tracking_lag_s
        self.over_since = (self.over_since or now) if error > allowed else None
        status = piper.latest_status()
        if any(flag in str(status.get("arm_status", "")) for flag in ("EXCEEDS_LIMIT", "NO_SOLUTION", "SINGULARITY")):
            return f"joint/workspace limit ({status.get('arm_status')}); joints status {status}"
        joints = piper.latest_joint_angles()
        if joints is not None:
            j5 = abs(math.degrees(joints[4]))
            margin = args.wrist_singularity_deg
            self.j5_floor = min(margin, j5) if self.j5_floor is None else max(self.j5_floor, min(margin, j5))
            if j5 < self.j5_floor - 0.5:
                return f"wrist singularity: J5 = {math.degrees(joints[4]):.1f} deg (J4/J6 would flip)"
        if self.over_since is not None and now - self.over_since >= args.tracking_error_hold_s:
            return f"tracking error {error:.1f} mm > {allowed:.1f} mm for {args.tracking_error_hold_s:g} s (obstacle?)"
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, required=True, help="quest_calibration.json")
    parser.add_argument("--replay", type=Path, help="Replay this rec_NN directory (or quest.csv) instead of recording first.")
    parser.add_argument("--quest-port", type=int, default=defaults.DEFAULT_QUEST_PORT)
    parser.add_argument("--quest-host", default=defaults.DEFAULT_QUEST_HOST)
    parser.add_argument("--dry-run", action="store_true", help="No arm: record and check, using the home FK pose.")
    parser.add_argument("--with-orientation", action="store_true", help="Also replay controller rotation.")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="1 = recorded speed, 0.5 = half speed.")
    parser.add_argument("--ease-s", type=float, default=0.5, help="Playback rate ramp at the start and end.")
    parser.add_argument("--home-joints-deg", type=float, nargs=6, default=None,
                        help="Default: config/teleop_home.json (see quest_teleop.py --set-home-here).")
    parser.add_argument("--home-speed-deg-s", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--max-offset-mm", type=float, default=150.0)
    parser.add_argument("--max-rotation-deg", type=float, default=30.0)
    parser.add_argument("--max-speed-mm-s", type=float, default=300.0)
    parser.add_argument("--max-angular-speed-deg-s", type=float, default=180.0)
    parser.add_argument("--approach-speed-mm-s", type=float, default=50.0)
    parser.add_argument("--approach-angular-speed-deg-s", type=float, default=30.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--start-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--max-gap-s", type=float, default=0.2)
    parser.add_argument("--quest-jump-mm", type=float, default=50.0)
    parser.add_argument("--quest-timeout-s", type=float, default=0.15)
    parser.add_argument("--max-tracking-error-mm", type=float, default=20.0)
    parser.add_argument("--tracking-lag-s", type=float, default=0.3)
    parser.add_argument("--tracking-error-hold-s", type=float, default=0.3)
    parser.add_argument("--wrist-singularity-deg", type=float, default=10.0)
    parser.add_argument("--no-pedal", action="store_true", help="Use Enter to start/stop recording.")
    parser.add_argument("--foot-pedal-device")
    parser.add_argument("--output-root", type=Path, default=Path("output/quest_replay"))
    parser.add_argument("--speed-percent", type=int, default=100)
    parser.add_argument("--interface", default=defaults.DEFAULT_PIPER_INTERFACE)
    parser.add_argument("--channel", default=defaults.DEFAULT_PIPER_CHANNEL)
    parser.add_argument("--firmware", choices=("default", "v183", "v188"), default=defaults.DEFAULT_PIPER_FIRMWARE)
    parser.add_argument("--installation-pos", choices=("unchanged", "horizontal", "left", "right"),
                        default=defaults.DEFAULT_PIPER_INSTALLATION_POS)
    parser.add_argument("--payload", choices=("unchanged", "empty", "half", "full"), default=defaults.DEFAULT_PIPER_PAYLOAD)
    args = parser.parse_args(argv)
    for name in ("playback_speed", "home_speed_deg_s", "rate_hz", "max_offset_mm", "max_rotation_deg",
                 "max_speed_mm_s", "max_angular_speed_deg_s", "approach_speed_mm_s", "approach_angular_speed_deg_s",
                 "start_tolerance_mm", "max_gap_s", "quest_jump_mm", "quest_timeout_s", "max_tracking_error_mm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("ease_s", "settle_s", "tracking_lag_s", "tracking_error_hold_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.playback_speed > 2:
        parser.error("--playback-speed is limited to 2")
    if args.home_joints_deg is None:
        args.home_joints_deg = list(saved_home_deg())
    return args


def run(args):
    for warning in home_warnings(args.home_joints_deg):
        print(f"WARNING: home {warning}; motion that pushes this joint further will stop at the limit.")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("tracker", "quest") != "quest":
        print(f"WARNING: {args.calibration} was fitted for tracker {calibration['tracker']!r}, not the Quest.")
    home = np.radians(args.home_joints_deg)
    session = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    commands_log, events = [], []
    limits = dict(max_offset_m=args.max_offset_mm / 1000, max_rotation_rad=math.radians(args.max_rotation_deg),
                  max_speed_m_s=args.max_speed_mm_s / 1000,
                  max_angular_speed_rad_s=math.radians(args.max_angular_speed_deg_s),
                  playback_speed=args.playback_speed, max_gap_s=args.max_gap_s, quest_jump_m=args.quest_jump_mm / 1000)

    def event(name, detail=""):
        events.append({"monotonic_ns": time.monotonic_ns(), "event": name, "detail": detail})
        print(f"[{name}] {detail}", flush=True)

    def next_key(timeout_s=None):
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while deadline is None or time.monotonic() < deadline:
            try:
                return keys.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    api = piper = pedal = quest = None
    keys = queue.Queue()
    replays = 0
    try:
        if args.dry_run:
            from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
            home_pose = fk_from_mdh(list(get_mdh("piper")), list(home))
        else:
            from scripts.piper_api.piper_api import PiperAPI
            api = PiperAPI(firmware=args.firmware, interface=args.interface, channel=args.channel,
                           installation_pos=args.installation_pos, payload=args.payload)
            api.connect()
            if not api.is_ok():
                raise RuntimeError("Piper is not responding on the CAN bus")
            input("Clear the workspace, keep a hand near the e-stop, and press Enter to move to home...")
            if not api.enable():
                raise RuntimeError("Timed out while enabling Piper")
            api.set_position_mode()
            time.sleep(POSITION_MODE_SETTLE_S)
            api.set_speed_percent(args.speed_percent)
            piper = PiperLogger(api, 200.0)
            piper.thread.start()
            go_home(api, home, args.home_speed_deg_s, args.rate_hz)
            home_pose = [float(value) for value in api.get_eef_pose()]
        event("home", f"pose {np.round(home_pose, 4).tolist()}")
        mapper = AbsoluteMapper(calibration, home_pose, with_orientation=args.with_orientation)
        if mapper.origin_spread > 0.005:
            print(f"WARNING: calibration recordings' origins differ by up to {1000 * mapper.origin_spread:.1f} mm")

        if args.replay is None:
            quest = QuestStream(args.quest_host, args.quest_port)
            deadline = time.monotonic() + defaults.DEFAULT_QUEST_STARTUP_TIMEOUT_S
            while quest.get() is None:
                if time.monotonic() > deadline:
                    raise RuntimeError(f"No Quest packets on UDP {args.quest_port}")
                time.sleep(0.05)
            print(f"Quest packets arriving on UDP {args.quest_port}.")
        if not args.no_pedal and (quest is not None or api is not None):
            from scripts.piper_api.foot_pedal import FootPedalTrigger
            pedal = FootPedalTrigger(args.foot_pedal_device, debounce_s=0.3)
            pedal.start()
        # Start reading keyboard commands only now: an earlier reader would swallow the Enter
        # that answers the "move to home" prompt above.
        threading.Thread(target=keyboard_reader, args=(keys,), daemon=True).start()

        def record():
            """Rows between two pedal presses (Enter with --no-pedal), or None if q was typed."""
            print(f"Arm holds still. Move the controller to the start pose, then "
                  f"{'press the pedal' if pedal else 'press Enter'} to start recording (q+Enter = quit).")
            if pedal:
                pedal.poll()  # drop presses made before this prompt
            start_ns, next_status = None, 0.0
            while True:
                pressed = bool(pedal.poll()) if pedal else False
                while not keys.empty():
                    key = keys.get()
                    if key == "q":
                        return None
                    pressed |= key == "" and pedal is None
                sample, now = quest.get(), time.monotonic()
                fresh = sample is not None and now - sample[0] <= args.quest_timeout_s
                if pressed and start_ns is None:
                    if fresh:
                        start_ns = time.monotonic_ns()
                        event("record_start")
                    else:
                        print("Cannot start: Quest data is stale", flush=True)
                elif pressed:
                    stop_ns = time.monotonic_ns()
                    event("record_stop", f"{(stop_ns - start_ns) / 1e9:.1f} s")
                    with quest.lock:
                        return [row for row in quest.rows if start_ns <= row["monotonic_ns"] <= stop_ns]
                if now >= next_status and sample is not None:
                    offset = 1000 * (mapper.map(sample[1], sample[2])[0][0] - mapper.home_p)
                    state = f"REC {(time.monotonic_ns() - start_ns) / 1e9:5.1f} s" if start_ns else "ready      "
                    flag = "OUT OF RANGE" if np.linalg.norm(offset) > args.max_offset_mm else "" if fresh else "stale"
                    print(f"{state}  target from home [{offset[0]:+7.1f} {offset[1]:+7.1f} {offset[2]:+7.1f}] mm "
                          f"({np.linalg.norm(offset):5.1f})  {flag}", flush=True)
                    next_status = now + STATUS_PERIOD_S
                time.sleep(0.02)

        def prepare(rows, directory, save):
            times, q, quat = quest_arrays(rows)
            trajectory = None
            if len(times) >= 2:
                trajectory = Trajectory(times, *mapper.map(q, quat))
            problems, stats = check_trajectory(trajectory, times, q, mapper, **limits)
            if save:
                directory.mkdir(parents=True, exist_ok=True)
                write_csv(directory / "quest.csv", QUEST_FIELDS, rows)
                if trajectory is not None:
                    write_csv(directory / "trajectory.csv", TRAJECTORY_FIELDS,
                              [dict(zip(TRAJECTORY_FIELDS, (f"{t:.6f}", *(f"{v:.9f}" for v in pose_list(p, r)))))
                               for t, p, r in zip(times, trajectory.positions, trajectory.rotations)])
            print(f"Recording {directory}: " + ", ".join(f"{key} {round(value, 1):g}" for key, value in stats.items()))
            for problem in problems:
                print(f"  NOT REPLAYABLE: {problem}")
            return trajectory, problems

        def stream(targets, command, guard, phase_log):
            """Send (phase, target) at --rate-hz, rate limited; returns (last command, stop reason or None)."""
            period = 1.0 / args.rate_hz
            max_step_m = args.max_speed_mm_s / 1000 * period
            max_step_rad = math.radians(args.max_angular_speed_deg_s) * period
            t0 = time.monotonic()
            for tick, (phase, goal) in enumerate(targets, start=1):
                if (pedal and pedal.poll()) or not keys.empty():
                    while not keys.empty():
                        keys.get()
                    return command, "stopped by operator"
                previous = command[0]
                command = rate_limit(command, goal, max_step_m, max_step_rad)
                pose = pose_list(*command)
                api.move_eef(pose)
                commands_log.append({"monotonic_ns": time.monotonic_ns(), "replay": replays, "phase": phase,
                                     **{name: f"{value:.9f}" for name, value in zip(COMMAND_FIELDS[3:], pose)}})
                fault = guard.check(piper, command[0], 1000 * np.linalg.norm(command[0] - previous) / period,
                                    time.monotonic())
                if fault:
                    measured = piper.latest_pose()
                    command = (np.asarray(measured[:3], float), Rotation.from_euler("xyz", measured[3:]))
                    api.move_eef(pose_list(*command))
                    return command, fault
                phase_log(phase, tick)
                sleep = t0 + tick * period - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
            return command, None

        def replay(trajectory):
            nonlocal replays
            replays += 1
            pose = api.get_eef_pose()
            command = (np.asarray(pose[:3], float), Rotation.from_euler("xyz", pose[3:]))
            start = trajectory.at(0.0)
            guard = ArmGuard(args)
            event("approach", f"replay {replays}: {1000 * np.linalg.norm(start[0] - command[0]):.0f} mm, "
                              f"{math.degrees((start[1] * command[1].inv()).magnitude()):.0f} deg to the start pose")
            settle = [("settle", start)] * max(1, round(args.settle_s * args.rate_hz))
            targets = [("approach", target) for target in approach_path(
                command, start, args.rate_hz, args.approach_speed_mm_s / 1000,
                math.radians(args.approach_angular_speed_deg_s))] + settle
            command, stop = stream(targets, command, guard, lambda phase, tick: None)
            if stop:
                event("stop", stop)
                return
            measured = piper.latest_pose()
            error = float("inf") if measured is None else 1000 * np.linalg.norm(np.asarray(measured[:3]) - start[0])
            if error > args.start_tolerance_mm:
                event("stop", f"start pose not reached ({error:.1f} mm > --start-tolerance-mm "
                              f"{args.start_tolerance_mm:g}); not replaying")
                return
            wall = replay_wall_s(trajectory.duration, args.playback_speed, args.ease_s)
            event("replay_start", f"replay {replays}: {wall:.1f} s, start error {error:.1f} mm")
            steps = math.ceil(wall * args.rate_hz)
            status_every = max(1, round(STATUS_PERIOD_S * args.rate_hz))

            def progress(phase, tick):
                if tick % status_every == 0:
                    print(f"replaying {tick / args.rate_hz:5.1f} / {wall:.1f} s", flush=True)
            targets = (("replay", trajectory.at(replay_clock(tick / args.rate_hz, trajectory.duration,
                                                             args.playback_speed, args.ease_s)))
                       for tick in range(1, steps + 1))
            command, stop = stream(targets, command, guard, progress)
            event("stop" if stop else "replay_end", stop or f"replay {replays}")

        rec_index = 0
        if args.replay is not None:
            source = args.replay / "quest.csv" if args.replay.is_dir() else args.replay
            with source.open(newline="", encoding="utf-8") as stream_file:
                rows = list(csv.DictReader(stream_file))
            trajectory, problems = prepare(rows, source.parent, save=False)
        else:
            trajectory, problems = None, None
        while True:
            if problems is None:
                if api is not None:
                    go_home(api, home, args.home_speed_deg_s, args.rate_hz)
                rows = record()
                if rows is None:
                    break
                rec_index += 1
                trajectory, problems = prepare(rows, session / f"rec_{rec_index:02d}", save=True)
            replay_help = "" if problems else ("Enter = check again (dry run), " if args.dry_run else "Enter = replay, ")
            print(f"{replay_help}{'r = record again, ' if quest else ''}h = home, q = home and quit", flush=True)
            key = next_key()
            if key == "" and not problems:
                if args.dry_run:
                    start = trajectory.at(0.0)
                    print(f"Would approach {1000 * np.linalg.norm(start[0] - mapper.home_p):.0f} mm from home, then "
                          f"replay {replay_wall_s(trajectory.duration, args.playback_speed, args.ease_s):.1f} s")
                else:
                    replay(trajectory)
            elif key == "r" and quest:
                problems = None
            elif key in ("h", "q"):
                if api is not None:
                    event("homing")
                    go_home(api, home, args.home_speed_deg_s, args.rate_hz)
                if key == "q":
                    break
        event("quit")
        return 0
    except KeyboardInterrupt:
        print("\nStopped; the arm holds its last target.")
        return 130
    finally:
        if pedal is not None:
            pedal.stop()
        if piper is not None:
            piper.stop()
        if quest is not None:
            quest.stop()
        if api is not None:
            api.disconnect()
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl-C must not truncate the log
        if commands_log or session.is_dir():
            session.mkdir(parents=True, exist_ok=True)
            (session / "meta.json").write_text(json.dumps(
                {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 "calibration": str(args.calibration)}, indent=2) + "\n")
            write_csv(session / "events.csv", EVENT_FIELDS, events)
            if commands_log:
                write_csv(session / "commands.csv", COMMAND_FIELDS, commands_log)
            if piper is not None:
                write_csv(session / "piper.csv", PIPER_FIELDS, piper.rows)
            print(f"Saved session log to {session}")


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
