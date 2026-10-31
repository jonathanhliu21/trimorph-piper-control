#!/usr/bin/env python3
"""Delta-based Quest -> Piper teleoperation from a fixed home pose.

    # Check the mapping with the arm untouched: prints targets while you move the controller.
    python scripts/teleop/quest_teleop.py --calibration CAL.json --quest-port 5005 --dry-run
    # Teleoperate (foot pedal held = follow, released = hold).
    python scripts/teleop/quest_teleop.py --calibration CAL.json --quest-port 5005

CAL.json is written by scripts/calib/fit_quest_calibration.py.

On start the arm moves in joint space to the fixed home configuration used by
quest_calibration_sequence.py (--home-joints-deg). Tracking is delta based:
when the clutch engages, the current Quest pose (q0, Rq0) and robot target
(p0, Rp0) become anchors, and every new Quest sample is mapped as

    position only (default):  p = p0 + scale * A (q - q0),  R = R_home   (tracked Quest point;
                              --lever-compensation adds (R_mapped - Rp0) l to follow the
                              calibrated flange-equivalent point instead)
    --with-orientation:       R = (B dRq B^T) Rp0, p = p0 + scale * A (q - q0): the robot rotates
                              about its flange. With --lever-compensation it instead uses
                              get_coord_transform.map_relative_pose (transform_chunk), i.e.
                              R = (B dRq B^T) Rp0,  p = p0 + scale * A (q - q0) + (R - Rp0) l

Releasing and re-engaging re-anchors at the last target, so the hand can be
repositioned without moving the arm (like lifting a mouse).

Clutch: foot pedal held = engaged (deadman; --clutch toggle makes each press
flip it). Without a pedal (--no-pedal) Enter toggles. Keyboard commands (type
then Enter): h = go home (while disengaged), q = go home and quit. Ctrl-C stops
streaming at once; the arm holds its last target and is not moved home.

Safety: targets are clamped to --max-offset-mm from home and --max-rotation-deg
from the home orientation, and rate limited to --max-speed-mm-s and
--max-angular-speed-deg-s. The clutch disengages when Quest data is older than
--quest-timeout-s, when one Quest step jumps more than --quest-jump-mm, or when
joint-feedback FK lags the target by more than --max-tracking-error-mm.
After such a fault the clutch must be released and pressed again to re-engage.
Quest latency from the calibration is not compensated (live control cannot use
future samples). Sessions are logged to --output-root/<timestamp>/.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import signal
import socket
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection.collection import config as defaults  # noqa: E402
from data_collection.collection.quest_worker import _decode_packet  # noqa: E402
from scripts.calib.get_coord_transform import map_relative_pose  # noqa: E402
from scripts.calib.quest_calibration_sequence import JOINT_STEP_LIMIT_RAD  # noqa: E402
from scripts.calib.rectangle_repeatability import (  # noqa: E402
    PIPER_FIELDS, POSITION_MODE_SETTLE_S, QUEST_FIELDS, PiperLogger, write_csv,
)

# Teleop home: arm pose chosen by the operator on 2026-10-31 (controller limits: J6 +-180 deg; J5 is
# 7.7 deg from its -70 deg limit, so large motions that bend the wrist further can hit it).
TELEOP_HOME_DEG = (-43.02, 113.46, -59.72, -51.58, -62.27, 128.05)
HOME_FILE = Path(__file__).resolve().parents[2] / "config" / "teleop_home.json"
# Controller-reported limits (deg) read from this arm on 2026-10-31; pyAgxArm's J6 preset (+-120) is narrower.
CONTROLLER_LIMITS_DEG = ((-150, 150), (0, 180), (-170, 0), (-100, 100), (-70, 70), (-180, 180))
COMMAND_FIELDS = ("monotonic_ns", "engaged", "x", "y", "z", "roll", "pitch", "yaw")
EVENT_FIELDS = ("monotonic_ns", "event", "detail")
STATUS_PERIOD_S = 0.5


class QuestStream:
    """Latest Quest pose from UDP, with every packet kept for the session log."""

    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((host, port))
        except OSError as exc:
            self.sock.close()
            raise RuntimeError(
                f"UDP port {port} is already in use ({exc.strerror}). Another program is receiving Quest "
                f"packets, often a teleop suspended with Ctrl-Z: run `jobs` and `kill -9 %1`, or find it with "
                f"`ss -ulpn | grep :{port}`.") from exc
        self.sock.settimeout(0.2)
        self.lock = threading.Lock()
        self.latest = None  # (monotonic_s, position[3], quaternion xyzw[4], seq)
        self.rows = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name="quest-udp", daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            try:
                data, _address = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            now_ns = time.monotonic_ns()
            try:
                packet = _decode_packet(data)
            except ValueError as exc:
                print(f"WARNING: [quest] {exc}", file=sys.stderr)
                continue
            position = np.array([packet["px"], packet["py"], packet["pz"]], dtype=float)
            quaternion = np.array([packet["qx"], packet["qy"], packet["qz"], packet["qw"]], dtype=float)
            norm = np.linalg.norm(quaternion)
            if norm < 0.5:
                continue
            with self.lock:
                self.latest = (now_ns / 1e9, position, quaternion / norm, packet["seq"])
                self.rows.append({"monotonic_ns": now_ns, **packet})

    def get(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sock.close()


class TeleopMapper:
    """Pure mapping + limits; no I/O, so it can be tested without hardware."""

    def __init__(self, calibration, home_pose, *, scale=1.0, with_orientation=False,
                 max_offset_m=0.1, max_rotation_rad=math.radians(45), compensate_lever=False):
        self.calibration = calibration
        self.a = np.asarray(calibration["linear_matrix"], dtype=float)
        self.b = np.asarray(calibration["orientation"]["rotation_matrix"], dtype=float)
        self.home_p = np.asarray(home_pose[:3], dtype=float)
        self.home_r = Rotation.from_euler("xyz", home_pose[3:])
        self.scale = scale
        self.with_orientation = with_orientation
        self.max_offset_m = max_offset_m
        self.max_rotation_rad = max_rotation_rad
        # Optional: follow the calibrated flange-equivalent point (fitted offset l, ~10 cm from the tracked
        # point) instead of the tracked point. Off by default: for a hand-held controller the tracked point
        # is near the grip, and the lever turns wrist rotation and orientation jitter into robot motion.
        self.compensate_lever = compensate_lever and "tracked_point_offset" in calibration
        self.anchor = None

    def engage(self, quest_position, quest_quaternion, robot_position, robot_rotation):
        self.anchor = (np.asarray(quest_position, float), Rotation.from_quat(quest_quaternion),
                       np.asarray(robot_position, float), robot_rotation)

    def disengage(self):
        self.anchor = None

    def target(self, quest_position, quest_quaternion):
        """Absolute (position, Rotation) target for the current Quest pose, clamped around home."""
        q0, rq0, p0, rp0 = self.anchor
        dq = np.asarray(quest_position, float) - q0
        if self.with_orientation and self.compensate_lever:
            delta_r = (Rotation.from_quat(quest_quaternion) * rq0.inv()).as_quat()
            dp, mapped = map_relative_pose(self.scale * dq, delta_r, self.calibration,
                                           piper_start_quaternion=rp0.as_quat())
            rotation = Rotation.from_quat(mapped) * rp0
        elif self.with_orientation:
            # Rotate about the flange: Dp = B dRq B^T (world-frame relative rotation), position from A only.
            delta = (Rotation.from_quat(quest_quaternion) * rq0.inv()).as_matrix()
            rotation = Rotation.from_matrix(self.b @ delta @ self.b.T) * rp0
            dp = self.scale * (self.a @ dq)
        elif self.compensate_lever:
            delta_r = (Rotation.from_quat(quest_quaternion) * rq0.inv()).as_quat()
            dp, _ = map_relative_pose(self.scale * dq, delta_r, self.calibration,
                                      piper_start_quaternion=rp0.as_quat())
            rotation = self.home_r
        else:
            dp = self.scale * (self.a @ dq)
            rotation = self.home_r
        position = p0 + dp
        offset = position - self.home_p
        if np.linalg.norm(offset) > self.max_offset_m:
            position = self.home_p + offset * (self.max_offset_m / np.linalg.norm(offset))
        away = rotation * self.home_r.inv()
        if away.magnitude() > self.max_rotation_rad:
            rotation = Rotation.from_rotvec(away.as_rotvec() * (self.max_rotation_rad / away.magnitude())) * self.home_r
        return position, rotation


def rate_limit(previous, target, max_step_m, max_step_rad):
    """Move from previous toward target by at most the given linear/angular step."""
    p_prev, r_prev = previous
    p_goal, r_goal = target
    step = p_goal - p_prev
    distance = np.linalg.norm(step)
    position = p_goal if distance <= max_step_m else p_prev + step * (max_step_m / distance)
    angle = (r_goal * r_prev.inv()).magnitude()
    if angle <= max_step_rad:
        rotation = r_goal
    else:
        rotation = Slerp([0, 1], Rotation.concatenate([r_prev, r_goal]))(max_step_rad / angle)
    return position, rotation


def pose_list(position, rotation):
    return [*map(float, position), *map(float, rotation.as_euler("xyz"))]


def saved_home_deg():
    try:
        return tuple(json.loads(HOME_FILE.read_text())["home_joints_deg"])
    except (OSError, ValueError, KeyError):
        return TELEOP_HOME_DEG


def home_warnings(joints_deg, margin_deg=15.0, wrist_deg=15.0):
    warnings = [f"J{joint} = {angle:.1f} deg is {min(angle - low, high - angle):.1f} deg from its limit ({low}..{high})"
                for joint, (angle, (low, high)) in enumerate(zip(joints_deg, CONTROLLER_LIMITS_DEG), start=1)
                if min(angle - low, high - angle) < margin_deg]
    if abs(joints_deg[4]) < wrist_deg:
        warnings.append(f"J5 = {joints_deg[4]:.1f} deg is near the wrist singularity (0)")
    return warnings


def set_home_here(args):
    """Read the arm's current joints (read-only) and save them as the teleop home."""
    from scripts.piper_api.piper_api import PiperAPI

    api = PiperAPI(firmware=args.firmware, interface=args.interface, channel=args.channel,
                   installation_pos="unchanged", payload="unchanged")
    api.robot.connect()
    time.sleep(0.5)
    try:
        joints = api.get_joint_angles()
        if joints is None:
            raise RuntimeError("No joint feedback; is the arm powered and CAN up?")
        joints_deg = [round(math.degrees(value), 2) for value in joints]
    finally:
        api.disconnect()
    HOME_FILE.parent.mkdir(parents=True, exist_ok=True)
    HOME_FILE.write_text(json.dumps({"home_joints_deg": joints_deg,
                                     "saved": datetime.now().isoformat(timespec="seconds")}, indent=2) + "\n")
    print(f"Saved teleop home {joints_deg} deg to {HOME_FILE}")
    for warning in home_warnings(joints_deg):
        print(f"WARNING: {warning}; teleop motion that pushes this joint further will stop at the limit.")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, help="quest_calibration.json (required unless --set-home-here)")
    parser.add_argument("--quest-port", type=int, default=defaults.DEFAULT_QUEST_PORT)
    parser.add_argument("--quest-host", default=defaults.DEFAULT_QUEST_HOST)
    parser.add_argument("--dry-run", action="store_true", help="No arm: print targets from the home FK pose.")
    parser.add_argument("--with-orientation", action="store_true", help="Also follow controller rotation.")
    parser.add_argument("--scale", type=float, default=1.0, help="Motion scale (robot mm per hand mm).")
    parser.add_argument("--lever-compensation", action="store_true",
                        help="Position-only: follow the calibrated flange-equivalent point (offset l) instead of the "
                             "tracked Quest point. Amplifies orientation jitter; off by default.")
    parser.add_argument("--home-joints-deg", type=float, nargs=6, default=None,
                        help="Default: config/teleop_home.json (see --set-home-here), else the built-in home.")
    parser.add_argument("--set-home-here", action="store_true",
                        help="Save the arm's current joint angles as the teleop home and exit (read-only).")
    parser.add_argument("--home-speed-deg-s", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--max-offset-mm", type=float, default=100.0)
    parser.add_argument("--max-rotation-deg", type=float, default=30.0)
    parser.add_argument("--max-speed-mm-s", type=float, default=300.0)
    parser.add_argument("--max-angular-speed-deg-s", type=float, default=180.0)
    parser.add_argument("--max-tracking-error-mm", type=float, default=20.0,
                        help="Static tracking allowance; speed * --tracking-lag-s is added while moving.")
    parser.add_argument("--tracking-lag-s", type=float, default=0.3,
                        help="Expected arm lag behind a moving target (measured ~0.14 s).")
    parser.add_argument("--tracking-error-hold-s", type=float, default=0.3,
                        help="Tracking error must exceed its allowance this long before disengaging.")
    parser.add_argument("--quest-timeout-s", type=float, default=0.15)
    parser.add_argument("--quest-jump-mm", type=float, default=50.0)
    parser.add_argument("--wrist-singularity-deg", type=float, default=10.0,
                        help="Disengage when |J5| comes within this of 0 (wrist singularity: J4/J6 flip).")
    parser.add_argument("--clutch", choices=("hold", "toggle"), default="hold")
    parser.add_argument("--no-pedal", action="store_true", help="Use Enter as a toggle clutch instead of the pedal.")
    parser.add_argument("--foot-pedal-device")
    parser.add_argument("--output-root", type=Path, default=Path("output/teleop"))
    parser.add_argument("--speed-percent", type=int, default=100)
    parser.add_argument("--interface", default=defaults.DEFAULT_PIPER_INTERFACE)
    parser.add_argument("--channel", default=defaults.DEFAULT_PIPER_CHANNEL)
    parser.add_argument("--firmware", choices=("default", "v183", "v188"), default=defaults.DEFAULT_PIPER_FIRMWARE)
    parser.add_argument("--installation-pos", choices=("unchanged", "horizontal", "left", "right"),
                        default=defaults.DEFAULT_PIPER_INSTALLATION_POS)
    parser.add_argument("--payload", choices=("unchanged", "empty", "half", "full"), default=defaults.DEFAULT_PIPER_PAYLOAD)
    args = parser.parse_args(argv)
    for name in ("scale", "home_speed_deg_s", "rate_hz", "max_offset_mm", "max_rotation_deg", "max_speed_mm_s",
                 "max_angular_speed_deg_s", "max_tracking_error_mm", "quest_timeout_s", "quest_jump_mm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("tracking_lag_s", "tracking_error_hold_s"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.home_joints_deg is None:
        args.home_joints_deg = list(saved_home_deg())
    if args.calibration is None and not args.set_home_here:
        parser.error("--calibration is required")
    if args.scale > 3:
        parser.error("--scale is limited to 3")
    return args


def go_home(api, home, speed_deg_s, rate_hz):
    """Joint-space ramp from measured joints to the fixed home configuration."""
    from scripts.piper_api.replay_trajectory import command_bounded_joint_step, read_joint_angles

    start = np.array(read_joint_angles(api))
    period = 1.0 / rate_hz
    steps = max(1, math.ceil(np.max(np.abs(home - start)) / (math.radians(speed_deg_s) * period)))
    t0 = time.monotonic()
    for tick in range(1, steps + int(rate_hz) + 1):  # + 1 s settle
        command_bounded_joint_step(api, start + (home - start) * min(tick / steps, 1.0), JOINT_STEP_LIMIT_RAD)
        sleep = t0 + tick * period - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
    error = math.degrees(np.max(np.abs(np.array(read_joint_angles(api)) - home)))
    if error > 1.0:
        raise RuntimeError(f"Arm did not reach home (worst joint error {error:.2f} deg)")


def keyboard_reader(commands):
    for line in sys.stdin:
        commands.put(line.strip().lower())


def run(args):
    if args.set_home_here:
        return set_home_here(args)
    for warning in home_warnings(args.home_joints_deg):
        print(f"WARNING: home {warning}; motion that pushes this joint further will stop at the limit.")
    calibration = json.loads(args.calibration.read_text())
    home = np.radians(args.home_joints_deg)
    session = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    commands_log, events = [], []

    def event(name, detail=""):
        events.append({"monotonic_ns": time.monotonic_ns(), "event": name, "detail": detail})
        print(f"[{name}] {detail}", flush=True)

    api = piper = pedal = None
    quest = QuestStream(args.quest_host, args.quest_port)
    keys = queue.Queue()
    try:
        deadline = time.monotonic() + defaults.DEFAULT_QUEST_STARTUP_TIMEOUT_S
        while quest.get() is None:
            if time.monotonic() > deadline:
                raise RuntimeError(f"No Quest packets on UDP {args.quest_port}")
            time.sleep(0.05)
        print(f"Quest packets arriving on UDP {args.quest_port}.")

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

        if not args.no_pedal:
            from scripts.piper_api.foot_pedal import FootPedalTrigger
            pedal = FootPedalTrigger(args.foot_pedal_device, debounce_s=0.0 if args.clutch == "hold" else 0.3)
            pedal.start()
        # Start reading keyboard commands only now: an earlier reader would swallow the Enter
        # that answers the "move to home" prompt above.
        threading.Thread(target=keyboard_reader, args=(keys,), daemon=True).start()
        clutch_help = ("hold the pedal to follow" if args.clutch == "hold" and pedal else
                       "press the pedal to toggle" if pedal else "press Enter to toggle")
        print(f"Ready: {clutch_help}. Type h+Enter = home, q+Enter = home and quit, Ctrl-C = stop.")

        mapper = TeleopMapper(calibration, home_pose, scale=args.scale, with_orientation=args.with_orientation,
                              max_offset_m=args.max_offset_mm / 1000,
                              max_rotation_rad=math.radians(args.max_rotation_deg),
                              compensate_lever=args.lever_compensation)
        period = 1.0 / args.rate_hz
        max_step_m = args.max_speed_mm_s / 1000 * period
        max_step_rad = math.radians(args.max_angular_speed_deg_s) * period
        command = (np.asarray(home_pose[:3], float), Rotation.from_euler("xyz", home_pose[3:]))
        toggled = False
        engaged = False
        latched = False  # set by a fault; cleared when the clutch is released
        over_since = None  # time tracking error first exceeded its allowance
        frozen_since = None  # Quest position unchanged while orientation changes (out of camera view)
        j5_floor = None  # wrist-singularity guard floor for the current engagement
        frozen_warned = False
        frozen_ref = None
        last_quest = None
        next_status = 0.0
        t0 = time.monotonic()
        tick = 0
        while True:
            # Inputs
            pressed = pedal.poll() if pedal else []
            quit_requested = home_requested = False
            while not keys.empty():
                key = keys.get()
                if key == "q":
                    quit_requested = True
                elif key == "h":
                    home_requested = True
                elif key == "" and pedal is None:
                    toggled = not toggled
            if pedal and args.clutch == "toggle" and pressed:
                toggled = not toggled
            want = (pedal.current_hold_s() is not None) if (pedal and args.clutch == "hold") else toggled

            sample = quest.get()
            now = time.monotonic()
            fresh = sample is not None and now - sample[0] <= args.quest_timeout_s
            jump = (last_quest is not None and sample is not None and sample[3] != last_quest[3]
                    and 1000 * np.linalg.norm(sample[1] - last_quest[1]) > args.quest_jump_mm)
            last_quest = sample
            if engaged and sample is not None:
                if frozen_ref is None or np.linalg.norm(sample[1] - frozen_ref[0]) > 1e-6:
                    frozen_ref, frozen_since = (sample[1].copy(), sample[2].copy()), now
                elif (not frozen_warned and now - frozen_since > 0.5 and
                      (Rotation.from_quat(sample[2]) * Rotation.from_quat(frozen_ref[1]).inv()).magnitude() > math.radians(3)):
                    event("warning", "Quest position is frozen while orientation changes: the headset cameras cannot "
                                     "see the controller, so translation is not tracked")
                    frozen_warned = True
            elif not engaged:
                frozen_ref, frozen_warned = None, False

            # Clutch state machine
            if not want:
                latched = False
            if want and not engaged and latched:
                if now >= next_status:
                    print("Fault latched: release the pedal and press again to re-engage", flush=True)
            elif want and not engaged:
                if not fresh:
                    if now >= next_status:
                        print("Cannot engage: Quest data is stale", flush=True)
                else:
                    mapper.engage(sample[1], sample[2], *command)
                    engaged = True
                    j5_floor = None
                    event("engage", f"anchor robot {np.round(1000 * command[0], 1).tolist()} mm")
            elif engaged and (not want or not fresh or jump):
                mapper.disengage()
                engaged = False
                reason = "released" if not want else "Quest stale" if not fresh else "Quest jump"
                latched = want
                event("disengage", reason)
                if args.clutch == "toggle" or pedal is None:
                    toggled = False

            if quit_requested or home_requested:
                if engaged:
                    mapper.disengage()
                    engaged = False
                    event("disengage", "home requested")
                if api is not None:
                    event("homing")
                    go_home(api, home, args.home_speed_deg_s, args.rate_hz)
                    command = (np.asarray(api.get_eef_pose()[:3], float), Rotation.from_euler("xyz", api.get_eef_pose()[3:]))
                else:
                    command = (np.asarray(home_pose[:3], float), Rotation.from_euler("xyz", home_pose[3:]))
                if quit_requested:
                    event("quit")
                    return 0
                t0, tick = time.monotonic(), 0
                continue

            # Target
            previous_position = command[0]
            if engaged:
                command = rate_limit(command, mapper.target(sample[1], sample[2]), max_step_m, max_step_rad)
            speed_mm_s = 1000 * np.linalg.norm(command[0] - previous_position) / period
            pose = pose_list(*command)
            if api is not None:
                api.move_eef(pose)
                measured = piper.latest_pose()
                if engaged and measured is not None:
                    # The arm trails a moving target by ~0.15 s, so allow speed * --tracking-lag-s on top of
                    # the static limit, and trip only if the excess persists (a stuck arm stays behind).
                    error = 1000 * np.linalg.norm(np.asarray(measured[:3]) - command[0])
                    allowed = args.max_tracking_error_mm + speed_mm_s * args.tracking_lag_s
                    status = piper.latest_status()
                    at_limit = any(flag in str(status.get("arm_status", ""))
                                   for flag in ("EXCEEDS_LIMIT", "NO_SOLUTION", "SINGULARITY"))
                    over_since = (over_since or now) if error > allowed else None
                    # Stop only motion toward the singularity: the floor starts at |J5| on engagement (so the
                    # arm can leave the zone after a stop) and ratchets up to the margin as the wrist opens.
                    joints_now = piper.latest_joint_angles()
                    near_singularity = False
                    if joints_now is not None:
                        j5 = abs(math.degrees(joints_now[4]))
                        j5_floor = min(args.wrist_singularity_deg, j5) if j5_floor is None else \
                            max(j5_floor, min(args.wrist_singularity_deg, j5))
                        near_singularity = j5 < j5_floor - 0.5
                    if near_singularity or at_limit or (over_since is not None
                                                        and now - over_since >= args.tracking_error_hold_s):
                        mapper.disengage()
                        engaged = False
                        toggled = False
                        latched = True
                        over_since = None
                        command = (np.asarray(measured[:3], float), Rotation.from_euler("xyz", measured[3:]))
                        reason = (f"wrist singularity: J5 = {math.degrees(joints_now[4]):.1f} deg (J4/J6 would flip); "
                                  "rotate the hand back toward the home orientation" if near_singularity else
                                  f"joint/workspace limit ({status.get('arm_status')}); move the hand back toward home"
                                  if at_limit else f"tracking error {error:.1f} mm > {allowed:.1f} mm for "
                                  f"{args.tracking_error_hold_s:g} s (obstacle?)")
                        event("disengage", f"{reason}; joints status {status}")
            commands_log.append({"monotonic_ns": time.monotonic_ns(), "engaged": int(engaged),
                                 **{name: f"{value:.9f}" for name, value in zip(COMMAND_FIELDS[2:], pose)}})
            if now >= next_status:
                offset = 1000 * (command[0] - np.asarray(home_pose[:3]))
                age = (now - sample[0]) * 1000 if sample else float("nan")
                print(f"{'ENGAGED ' if engaged else 'holding '} offset from home "
                      f"[{offset[0]:+7.1f} {offset[1]:+7.1f} {offset[2]:+7.1f}] mm  quest age {age:4.0f} ms",
                      flush=True)
                next_status = now + STATUS_PERIOD_S

            tick += 1
            sleep = t0 + tick * period - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nStopped; the arm holds its last target.")
        return 130
    finally:
        if pedal is not None:
            pedal.stop()
        if piper is not None:
            piper.stop()
        quest.stop()
        if api is not None:
            api.disconnect()
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl-C must not truncate the log
        if commands_log:
            session.mkdir(parents=True, exist_ok=True)
            (session / "meta.json").write_text(json.dumps(
                {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 "calibration": str(args.calibration)}, indent=2) + "\n")
            write_csv(session / "commands.csv", COMMAND_FIELDS, commands_log)
            write_csv(session / "events.csv", EVENT_FIELDS, events)
            write_csv(session / "quest.csv", QUEST_FIELDS, quest.rows)
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
