#!/usr/bin/env python3
"""Drive Piper around a small rectangle repeatedly while logging FK, Vicon and Quest.

Run 1 (Vicon marker on the arm)::

    python scripts/calib/rectangle_repeatability.py --label mocap

Run 2 (Quest controller on the arm; Quest sender streaming to --quest-port)::

    python scripts/calib/rectangle_repeatability.py --label quest --quest-port 5005

Then: ``python scripts/calib/analyze_rectangle.py output/rect_repeatability/<run>``.

The arm first moves in joint space to --start-joints-deg (default: the home
used by quest_calibration_sequence.py), or stays put with --start-here; that
flange pose is the rectangle's first corner. Before anything moves, IK is
solved along the whole rectangle from the start joints and a per-joint table
shows each joint's range and its distance to the limits and to the wrist
singularity (J5 = 0); the run is refused if any is too small. --check-only
prints that table and exits (no hardware needed unless --start-here).

The rectangle lies in --plane of the Piper base frame; orientation is held at
the start orientation throughout. Each loop traverses the four edges at --speed-mm-s,
dwelling --dwell-s at every corner. Targets are streamed with move_eef at
--rate-hz exactly like replay_trajectory.py, so edges are straight lines in
the commanded trajectory. Streaming stops (the arm holds its last target) if
the Piper-reported flange pose deviates from the commanded one by more than
--max-tracking-error-mm.

Piper FK is computed from measured joint angles with pyAgxArm's modified-DH
model (robot.fk; flange, meters/radians) and logged as fk_*; the
firmware-reported flange pose is logged alongside as fw_* for comparison. The
rectangle itself is commanded in the firmware pose frame (move_eef). Vicon is read through scripts/calib/mocap_stream.py
under .venv-mocap with ros2/mocap_env.sh sourced (Foxy's rclpy is Python 3.8).
All streams carry host time.monotonic_ns() receipt stamps. Output, in
--output-root/<timestamp>_<label>/: piper.csv, commands.csv, mocap.csv,
quest.csv (with --quest-port) and meta.json. Partial data is saved on Ctrl-C
or failure. The arm is left enabled, holding its final pose.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection.collection import config as defaults  # noqa: E402
from data_collection.collection.piper_worker import piper_workspace_warning  # noqa: E402
from data_collection.collection.quest_worker import _decode_packet  # noqa: E402
from scripts.piper_api.piper_api import PiperAPI  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
# Last good pose from the 2026-10-31 rectangle runs (J5 well clear of 0 and its limits).
DEFAULT_HOME_DEG = (2.4, 161.4, -91.8, -8.1, 25.6, 105.0)
JOINT_STEP_LIMIT_RAD = 0.05
START_TOLERANCE_DEG = 1.0
START_SETTLE_S = 1.0
IK_PATH_STEP_M = 0.001
POSITION_MODE_SETTLE_S = 0.5
MAX_RECTANGLE_MM = 100.0
MOCAP_STALE_S = 1.0

FK_FIELDS = tuple(f"fk_{name}" for name in defaults.RAW_EEF_FIELDS)
FW_FIELDS = tuple(f"fw_{name}" for name in defaults.RAW_EEF_FIELDS)
STATUS_FIELDS = ("ctrl_mode", "arm_status", "mode_feedback", "motion_status", "err_code")
STATUS_PERIOD_S = 0.1
PIPER_FIELDS = ("monotonic_ns", "joint_timestamp_s", *(f"joint_{joint}" for joint in range(1, 7)),
                *FK_FIELDS, "fw_timestamp_s", *FW_FIELDS, *STATUS_FIELDS)
FK_FRAME_WARNING_MM = 5.0
COMMAND_FIELDS = ("monotonic_ns", "loop", "phase", "edge", "corner", "x", "y", "z", "roll", "pitch", "yaw")
MOCAP_FIELDS = ("monotonic_ns", "frame_number", "x", "y", "z", "qx", "qy", "qz", "qw", "valid")
QUEST_FIELDS = ("monotonic_ns", *defaults.QUEST_PACKET_FIELDS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="run", help="Suffix for the output directory name.")
    parser.add_argument("--output-root", type=Path, default=Path("output/rect_repeatability"))
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--width-mm", type=float, default=20.0, help="Edge along the plane's first axis.")
    parser.add_argument("--height-mm", type=float, default=20.0, help="Edge along the plane's second axis.")
    parser.add_argument("--plane", choices=sorted(PLANES), default="xy",
                        help="Piper base-frame plane of the rectangle (default: xy, horizontal).")
    parser.add_argument("--speed-mm-s", type=float, default=10.0)
    parser.add_argument("--dwell-s", type=float, default=1.0, help="Hold time at each corner.")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Command streaming rate.")
    parser.add_argument("--log-rate-hz", type=float, default=200.0, help="Piper state logging rate.")
    parser.add_argument("--max-tracking-error-mm", type=float, default=10.0)
    parser.add_argument("--speed-percent", type=int, default=100)
    parser.add_argument("--joint-margin-deg", type=float, default=10.0,
                        help="Refuse to start unless every joint stays this far from its limits over the rectangle.")
    parser.add_argument("--wrist-singularity-margin-deg", type=float, default=15.0,
                        help="Refuse to start unless |J5| stays at least this large (wrist singularity at 0).")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--start-joints-deg", type=float, nargs=6, default=DEFAULT_HOME_DEG, metavar="DEG",
                       help="Joint configuration (J1..J6, deg) to move to in joint space before the rectangle; "
                            "it becomes corner 0 (default: %(default)s).")
    start.add_argument("--start-here", action="store_true",
                       help="Use the arm's current pose as corner 0 instead of moving to --start-joints-deg.")
    parser.add_argument("--start-speed-deg-s", type=float, default=10.0,
                        help="Joint-space speed of the move to --start-joints-deg.")
    parser.add_argument("--check-only", action="store_true",
                        help="Print the joint margin table for the start pose and rectangle, then exit "
                             "(no motion; with --start-here it reads the arm's current joints).")
    parser.add_argument("--mocap-topic", default="/vicon/Yori/Yori")
    parser.add_argument("--no-mocap", action="store_true", help="Do not start the Vicon relay.")
    parser.add_argument("--quest-port", type=int, help="Also record Quest UDP packets on this port.")
    parser.add_argument("--quest-host", default=defaults.DEFAULT_QUEST_HOST)
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    parser.add_argument("--interface", default=defaults.DEFAULT_PIPER_INTERFACE)
    parser.add_argument("--channel", default=defaults.DEFAULT_PIPER_CHANNEL)
    parser.add_argument("--firmware", choices=("default", "v183", "v188"), default=defaults.DEFAULT_PIPER_FIRMWARE)
    parser.add_argument("--installation-pos", choices=("unchanged", "horizontal", "left", "right"),
                        default=defaults.DEFAULT_PIPER_INSTALLATION_POS)
    parser.add_argument("--payload", choices=("unchanged", "empty", "half", "full"),
                        default=defaults.DEFAULT_PIPER_PAYLOAD)
    args = parser.parse_args(argv)
    if args.loops < 1:
        parser.error("--loops must be at least 1")
    for name in ("width_mm", "height_mm", "speed_mm_s", "rate_hz", "log_rate_hz", "max_tracking_error_mm",
                 "start_speed_deg_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.dwell_s) or args.dwell_s < 0:
        parser.error("--dwell-s must be finite and non-negative")
    if max(args.width_mm, args.height_mm) > MAX_RECTANGLE_MM:
        parser.error(f"rectangle edges are limited to {MAX_RECTANGLE_MM:g} mm")
    if not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be between 1 and 100")
    return args


def rectangle_corners(start_xyz, plane, width_m, height_m):
    """Corner 0 is the start; corners proceed +u, +v, -u, -v."""
    u, v = PLANES[plane]
    corners = []
    for du, dv in ((0, 0), (width_m, 0), (width_m, height_m), (0, height_m)):
        corner = list(start_xyz)
        corner[u] += du
        corner[v] += dv
        corners.append(corner)
    return corners


def build_schedule(corners, loops, speed_m_s, dwell_s):
    """Return [(t_start, t_end, loop, phase, edge, corner, from_xyz, to_xyz)]; corner 0 dwell first."""
    schedule = []
    t = 0.0
    schedule.append((t, t + dwell_s, 0, "dwell", -1, 0, corners[0], corners[0]))
    t += dwell_s
    for loop in range(loops):
        for edge in range(4):
            start, end = corners[edge], corners[(edge + 1) % 4]
            duration = math.dist(start, end) / speed_m_s
            schedule.append((t, t + duration, loop, "move", edge, -1, start, end))
            t += duration
            schedule.append((t, t + dwell_s, loop, "dwell", -1, (edge + 1) % 4, end, end))
            t += dwell_s
    return schedule


def target_at(segment, t):
    t0, t1, _loop, _phase, _edge, _corner, start, end = segment
    alpha = 1.0 if t1 <= t0 else min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return [a + alpha * (b - a) for a, b in zip(start, end)]


def piper_fk(joints):
    """Flange pose [m, rad] from joint angles, offline; the same modified-DH model as robot.fk."""
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh

    return np.array(fk_from_mdh(get_mdh("piper"), [float(angle) for angle in joints]))


def pose_error(pose, target):
    """[dx, dy, dz, rotation vector] taking pose to target (Euler xyz, Rz @ Ry @ Rx)."""
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler("xyz", target[3:]) * Rotation.from_euler("xyz", pose[3:]).inv()
    return np.concatenate([np.subtract(target[:3], pose[:3]), rotation.as_rotvec()])


def ik_track(joints, target, iterations=100, damping=1e-3, step_rad=1e-6):
    """Damped least-squares IK started at joints, so it stays on the arm's current branch."""
    q = np.array(joints, dtype=float)
    for _ in range(iterations):
        pose = piper_fk(q)
        error = pose_error(pose, target)
        if np.linalg.norm(error[:3]) < 1e-7 and np.linalg.norm(error[3:]) < 1e-6:
            return q
        jacobian = np.column_stack([pose_error(pose, piper_fk(q + step_rad * np.eye(6)[k])) / step_rad
                                    for k in range(6)])
        q = q + jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(6), error)
    raise RuntimeError(f"IK did not converge for target {np.round(target, 4).tolist()} "
                       "(out of reach or at a singularity)")


def rectangle_joint_path(start_joints, corners, orientation):
    """Joint angles every IK_PATH_STEP_M along the four edges, tracked continuously from start_joints."""
    q = np.array(start_joints, dtype=float)
    path = [q]
    for edge in range(4):
        start, end = np.array(corners[edge]), np.array(corners[(edge + 1) % 4])
        steps = max(1, math.ceil(np.linalg.norm(end - start) / IK_PATH_STEP_M))
        for step in range(1, steps + 1):
            q = ik_track(q, np.concatenate([start + (end - start) * step / steps, orientation]))
            path.append(q)
    return np.array(path)


def joint_margins(path, margin_deg, wrist_margin_deg):
    """Per-joint range over the path and its worst limit margin; list of problems (empty if OK)."""
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD

    limits = ROBOT_JOINT_LIMIT_PRESET_RAD["piper"]
    rows, problems = [], []
    for joint in range(6):
        low, high = (math.degrees(value) for value in limits[f"joint{joint + 1}"])
        values = np.degrees(path[:, joint])
        margin = min(values.min() - low, high - values.max())
        rows.append({"joint": joint + 1, "start_deg": values[0], "min_deg": values.min(), "max_deg": values.max(),
                     "low_deg": low, "high_deg": high, "margin_deg": margin})
        if margin < margin_deg:
            problems.append(f"J{joint + 1} comes within {margin:.1f} deg of its limits "
                            f"({low:.0f}..{high:.0f}; reaches {values.min():.1f}..{values.max():.1f})")
    # J4 and J6 align at J5 = 0, so small Cartesian moves need large opposite J4/J6 swings.
    wrist_deg = float(np.degrees(np.abs(path[:, 4]).min()))
    if wrist_deg < wrist_margin_deg:
        problems.append(f"|J5| drops to {wrist_deg:.1f} deg, within {wrist_margin_deg:g} deg of the wrist singularity (0)")
    return rows, wrist_deg, problems


def print_margins(rows, wrist_deg, margin_deg, wrist_margin_deg):
    print(f"Joint margins over the rectangle (need >= {margin_deg:g} deg from limits, |J5| >= {wrist_margin_deg:g} deg):")
    print(f"  {'joint':>5} {'start':>7} {'min':>7} {'max':>7} {'limits':>13} {'margin':>7}")
    for row in rows:
        flag = "  <-- too close" if row["margin_deg"] < margin_deg else ""
        print(f"  {'J' + str(row['joint']):>5} {row['start_deg']:7.1f} {row['min_deg']:7.1f} {row['max_deg']:7.1f} "
              f"{row['low_deg']:6.0f}..{row['high_deg']:<5.0f} {row['margin_deg']:7.1f}{flag}")
    flag = "  <-- near wrist singularity" if wrist_deg < wrist_margin_deg else ""
    print(f"  smallest |J5| over the rectangle: {wrist_deg:.1f} deg{flag}")


def check_rectangle_joints(start_joints, args):
    """Predict joints over the rectangle from start_joints, print the margin table, raise if too close."""
    start = piper_fk(start_joints)
    corners = rectangle_corners(start[:3], args.plane, args.width_mm / 1000, args.height_mm / 1000)
    rows, wrist_deg, problems = joint_margins(rectangle_joint_path(start_joints, corners, start[3:]),
                                              args.joint_margin_deg, args.wrist_singularity_margin_deg)
    print(f"Start joints (deg): {np.round(np.degrees(start_joints), 1).tolist()}; "
          f"flange (m, rad): {np.round(start, 4).tolist()}")
    print_margins(rows, wrist_deg, args.joint_margin_deg, args.wrist_singularity_margin_deg)
    if problems:
        raise RuntimeError("Start pose is too close to joint limits or the wrist singularity: " + "; ".join(problems)
                           + ". Choose other --start-joints-deg (or a smaller rectangle / other --plane); "
                           "--joint-margin-deg and --wrist-singularity-margin-deg set the thresholds.")


def move_to_joints(api, target, speed_deg_s, rate_hz):
    """Joint-space ramp from the measured joints to target at speed_deg_s, then verify."""
    from scripts.piper_api.replay_trajectory import command_bounded_joint_step, read_joint_angles

    start = np.array(read_joint_angles(api))
    period = 1.0 / rate_hz
    steps = max(1, math.ceil(np.max(np.abs(target - start)) / (math.radians(speed_deg_s) * period)))
    t0 = time.monotonic()
    for tick in range(1, steps + int(START_SETTLE_S * rate_hz) + 1):
        command_bounded_joint_step(api, start + (target - start) * min(tick / steps, 1.0), JOINT_STEP_LIMIT_RAD)
        sleep_s = t0 + tick * period - time.monotonic()
        if sleep_s > 0:
            time.sleep(sleep_s)
    error = math.degrees(np.max(np.abs(np.array(read_joint_angles(api)) - target)))
    if error > START_TOLERANCE_DEG:
        raise RuntimeError(f"Arm did not reach the start joints (worst joint error {error:.2f} deg)")


def joint_fk(api, joints):
    """Flange pose [m, rad] from measured joint angles via pyAgxArm's modified-DH model."""
    return [float(value) for value in api.robot.fk([float(angle) for angle in joints])]


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class PiperLogger:
    def __init__(self, api, rate_hz):
        self.api = api
        self.period_s = 1.0 / rate_hz
        self.rows = []
        self.latest = None
        self.latest_joints = None
        self.status = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, name="piper-log", daemon=True)

    def run(self):
        try:
            next_t = time.monotonic()
            next_status_t = next_t
            last_warning = None
            while not self.stop_event.is_set():
                if time.monotonic() >= next_status_t:
                    state = self.api.get_state()
                    warning = piper_workspace_warning(state)
                    if warning is not None and warning != last_warning:
                        print(warning, flush=True)
                    last_warning = warning
                    with self.lock:
                        self.status = {field: state.get(field, "") for field in STATUS_FIELDS}
                    next_status_t += STATUS_PERIOD_S
                joints = self.api.get_joint_state()
                eef = self.api.get_eef_state()
                sample_ns = time.monotonic_ns()
                if joints is not None:
                    fk = joint_fk(self.api, joints.joints)
                    row = {"monotonic_ns": sample_ns, "joint_timestamp_s": f"{joints.timestamp:.6f}"}
                    row.update({f"joint_{joint}": f"{float(value):.9f}" for joint, value in
                                enumerate(joints.joints, start=1)})
                    row.update({name: f"{value:.9f}" for name, value in zip(FK_FIELDS, fk)})
                    if eef is not None:
                        row["fw_timestamp_s"] = f"{eef.timestamp:.6f}"
                        row.update({name: f"{float(value):.9f}" for name, value in zip(FW_FIELDS, eef.pose)})
                    with self.lock:
                        row.update(self.status)
                        self.rows.append(row)
                        self.latest = fk
                        self.latest_joints = [float(angle) for angle in joints.joints]
                next_t += self.period_s
                self.stop_event.wait(max(0.0, next_t - time.monotonic()))
        except Exception as exc:
            self.error = exc

    def latest_pose(self):
        with self.lock:
            return self.latest

    def latest_joint_angles(self):
        with self.lock:
            return self.latest_joints

    def latest_status(self):
        with self.lock:
            return dict(self.status)

    def stop(self):
        self.stop_event.set()
        self.thread.join()


class MocapRelay:
    """Run mocap_stream.py under the Foxy venv and collect its JSON lines."""

    def __init__(self, topic):
        self.rows = []
        self.lock = threading.Lock()
        self.last_valid_ns = None
        self.last_ns = None
        # No positional args: a sourced script inherits the caller's $1 as its interface name.
        script = ('source ros2/mocap_env.sh >&2 && '
                  'exec .venv-mocap/bin/python -u scripts/calib/mocap_stream.py --mocap-topic "$MOCAP_TOPIC"')
        self.process = subprocess.Popen(["bash", "-c", script], cwd=REPO, stdout=subprocess.PIPE, text=True,
                                        env={**os.environ, "MOCAP_TOPIC": topic})
        self.thread = threading.Thread(target=self.run, name="mocap-relay", daemon=True)
        self.thread.start()

    def run(self):
        for line in self.process.stdout:
            sample = json.loads(line)
            position = (sample["x"], sample["y"], sample["z"])
            quaternion = (sample["qx"], sample["qy"], sample["qz"], sample["qw"])
            valid = any(position) and any(quaternion)
            sample["valid"] = int(valid)
            with self.lock:
                self.rows.append(sample)
                self.last_ns = sample["monotonic_ns"]
                if valid:
                    self.last_valid_ns = sample["monotonic_ns"]

    def wait_ready(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("mocap_stream.py exited; see its messages above")
            with self.lock:
                if self.last_valid_ns is not None:
                    return
                receiving = self.last_ns is not None
            time.sleep(0.1)
        if receiving:
            raise RuntimeError("Vicon samples arrive but the object pose is all zeros: "
                               "Vicon has lost the object (occluded or disabled in Tracker)")
        raise RuntimeError(f"No Vicon samples within {timeout_s:g}s; check the stream "
                           "(ros2/README.md, 'Check remote messages')")

    def stale_s(self):
        with self.lock:
            last = self.last_valid_ns
        return math.inf if last is None else (time.monotonic_ns() - last) / 1e9

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.thread.join(timeout=5)


class QuestReceiver:
    def __init__(self, host, port):
        self.rows = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((host, port))
        except OSError as exc:
            self.sock.close()
            hint = " (ports below 1024 need root; the Quest sender uses 5005)" if port < 1024 else \
                " (is another receiver, e.g. teleop or a second recorder, already using it?)"
            raise RuntimeError(f"Cannot listen for Quest packets on UDP {host}:{port}: {exc.strerror}{hint}") from exc
        self.sock.settimeout(0.2)
        self.thread = threading.Thread(target=self.run, name="quest-udp", daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            try:
                data, _address = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            sample_ns = time.monotonic_ns()
            try:
                packet = _decode_packet(data)
            except ValueError as exc:
                print(f"WARNING: [quest] {exc}", file=sys.stderr)
                continue
            with self.lock:
                self.rows.append({"monotonic_ns": sample_ns, **packet})

    def wait_ready(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self.lock:
                if self.rows:
                    return
            time.sleep(0.1)
        raise RuntimeError(f"No Quest UDP packets on port {self.sock.getsockname()[1]} within {timeout_s:g}s")

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sock.close()


def save(run_dir, meta, piper, commands, mocap, quest):
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "piper.csv", PIPER_FIELDS, piper.rows if piper else [])
    write_csv(run_dir / "commands.csv", COMMAND_FIELDS, commands)
    if mocap is not None:
        write_csv(run_dir / "mocap.csv", MOCAP_FIELDS, mocap.rows)
    if quest is not None:
        write_csv(run_dir / "quest.csv", QUEST_FIELDS, quest.rows)
    meta["counts"] = {"piper": len(piper.rows) if piper else 0, "commands": len(commands),
                      "mocap": len(mocap.rows) if mocap else None, "quest": len(quest.rows) if quest else None}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved {run_dir} ({meta['counts']})")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args):
    run_dir = args.output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.label}"
    meta = {"args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "git_commit": git_commit(), "status": "started",
            "units": {"piper": "m, rad (flange pose)", "mocap": "mm (Vicon global)", "quest": "native"}}
    commands = []
    api = mocap = quest = piper = None
    try:
        if not args.start_here:
            # Planned start joints: checkable before touching any hardware.
            check_rectangle_joints(np.radians(args.start_joints_deg), args)
            if args.check_only:
                return 0
        if not args.check_only and not args.no_mocap:
            print("Starting Vicon relay...")
            mocap = MocapRelay(args.mocap_topic)
            mocap.wait_ready(20.0)
            print("Vicon object tracked.")
        if not args.check_only and args.quest_port is not None:
            quest = QuestReceiver(args.quest_host, args.quest_port)
            quest.wait_ready(defaults.DEFAULT_QUEST_STARTUP_TIMEOUT_S)
            print(f"Quest packets arriving on UDP {args.quest_port}.")

        api = PiperAPI(firmware=args.firmware, interface=args.interface, channel=args.channel,
                       installation_pos=args.installation_pos, payload=args.payload)
        api.connect()
        print(f"Piper connected: {api.is_ok()} (receive FPS: {api.get_fps():.1f})")
        if not api.is_ok():
            raise RuntimeError("Piper is not responding on the CAN bus")
        current = api.get_joint_angles()
        if current is None:
            raise RuntimeError("Piper joint feedback unavailable")
        current = np.array([float(angle) for angle in current])
        if args.start_here:
            check_rectangle_joints(current, args)
            if args.check_only:
                return 0
        else:
            print(f"Current joints (deg): {np.round(np.degrees(current), 1).tolist()}; largest move to the start "
                  f"joints {np.degrees(np.max(np.abs(np.radians(args.start_joints_deg) - current))):.1f} deg "
                  f"at {args.start_speed_deg_s:g} deg/s")
        print(f"{args.loops} loops of {args.width_mm:g} x {args.height_mm:g} mm in the {args.plane} plane, "
              f"{args.speed_mm_s:g} mm/s, {args.dwell_s:g} s dwell.")
        if not args.yes:
            action = "start" if args.start_here else "move to the start joints and start"
            input(f"Clear the workspace, keep a hand near the e-stop, and press Enter to {action} (Ctrl-C aborts)...")

        if not api.enable():
            raise RuntimeError("Timed out while enabling Piper")
        api.set_position_mode()
        time.sleep(POSITION_MODE_SETTLE_S)
        api.set_speed_percent(args.speed_percent)
        if not args.start_here:
            move_to_joints(api, np.radians(args.start_joints_deg), args.start_speed_deg_s, args.rate_hz)
            print("At the start joints.")

        # move_eef targets are in the firmware's pose frame, so the rectangle is built
        # from the firmware-reported pose; joint FK is what gets logged and checked.
        start = api.get_eef_pose()
        start_joints = api.get_joint_angles()
        if start is None or not all(math.isfinite(value) for value in start) or math.hypot(*start[:3]) > 1.5:
            raise RuntimeError(f"Implausible Piper flange pose: {start}")
        if start_joints is None:
            raise RuntimeError("Piper joint feedback unavailable")
        start_fk = joint_fk(api, start_joints)
        frame_gap_mm = 1000 * math.dist(start[:3], start_fk[:3])
        print(f"Joint FK vs firmware flange pose at start: {frame_gap_mm:.2f} mm")
        if frame_gap_mm > FK_FRAME_WARNING_MM:
            print(f"WARNING: joint FK and firmware pose differ by {frame_gap_mm:.1f} mm; "
                  "check --installation-pos and the FK model before trusting FK-vs-commanded numbers")

        corners = rectangle_corners(start[:3], args.plane, args.width_mm / 1000, args.height_mm / 1000)
        schedule = build_schedule(corners, args.loops, args.speed_mm_s / 1000, args.dwell_s)
        total_s = schedule[-1][1]
        meta.update(start_pose=start, start_joints=[float(angle) for angle in start_joints], start_fk=start_fk,
                    corners=corners, plane=args.plane, schedule_duration_s=total_s)
        print(f"Start flange pose (m, rad): {[round(value, 4) for value in start]}")
        for index, corner in enumerate(corners):
            print(f"  corner {index}: {[round(value, 4) for value in corner]}")
        print(f"Running for {total_s:.0f} s.")

        piper = PiperLogger(api, args.log_rate_hz)
        piper.thread.start()
        period_s = 1.0 / args.rate_hz
        orientation = list(start[3:])
        segment_index = 0
        reported_loop = -1
        mocap_warned = False
        t0 = time.monotonic()
        tick = 0
        while True:
            t = tick * period_s
            if t > total_s:
                break
            while schedule[segment_index][1] < t and segment_index < len(schedule) - 1:
                segment_index += 1
            segment = schedule[segment_index]
            target = target_at(segment, t) + orientation
            api.move_eef(target)
            commands.append({"monotonic_ns": time.monotonic_ns(), "loop": segment[2], "phase": segment[3],
                             "edge": segment[4], "corner": segment[5],
                             **{name: f"{value:.9f}" for name, value in zip(defaults.RAW_EEF_FIELDS, target)}})

            if piper.error is not None:
                raise RuntimeError(f"Piper logging failed: {piper.error}") from piper.error
            measured = piper.latest_pose()
            if measured is not None:
                # Compare displacements so a constant FK/firmware frame offset cannot trip the check.
                error_mm = 1000 * math.dist(np.subtract(measured[:3], start_fk[:3]),
                                            np.subtract(target[:3], start[:3]))
                if error_mm > args.max_tracking_error_mm:
                    raise RuntimeError(f"Joint-FK flange is {error_mm:.1f} mm from the commanded target "
                                       f"(limit {args.max_tracking_error_mm:g} mm); stopping. "
                                       f"Piper status: {piper.latest_status()}")
            if mocap is not None:
                stale = mocap.stale_s()
                if stale > MOCAP_STALE_S and not mocap_warned:
                    print(f"WARNING: no valid Vicon pose for {stale:.1f}s (lost object or dropped stream)",
                          flush=True)
                    mocap_warned = True
                elif stale <= MOCAP_STALE_S:
                    mocap_warned = False
            if segment[3] == "dwell" and segment[5] == 0 and segment[2] != reported_loop and segment_index > 0:
                reported_loop = segment[2]
                print(f"Loop {reported_loop + 1}/{args.loops} done ({time.monotonic() - t0:.0f} s)", flush=True)

            tick += 1
            sleep_s = t0 + tick * period_s - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
        meta["status"] = "complete"
        print("Rectangle runs complete.")
        return 0
    except KeyboardInterrupt:
        meta["status"] = "interrupted"
        print("\nInterrupted; the arm holds its last commanded target.")
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
        if commands:
            save(run_dir, meta, piper, commands, mocap, quest)


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
