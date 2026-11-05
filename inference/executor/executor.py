"""Execute 19D policy chunks on Trimorph + Piper, using collection coordinates.

By default the request/reply loop is synchronous: prediction latency remains a
pause between chunks, and the timings expose that cost. --async-inference
overlaps prediction with playback and replaces the old tail at an action boundary.
It is mutually exclusive with --rtc, which runs real-time
chunking instead (run_rtc, inference/predictor/rtc.py): the next chunk is
requested while the current one executes and is guided to continue it, so the
arm does not stop between chunks. Before the first observation the arm ramps
in joint space to the
named start --start-home (default home_4c, from collection config.py) or
--start-joints-deg or a logged --start-id, optionally offset by a random flange
pose change (--start-jitter-mm / --start-jitter-deg, off by default), then the
hand moves to that start's median demo hand pose. Every start reached is
appended to output/executor_starts.jsonl with an id that --start-id repeats;
--no-start-move skips both. While the policy runs, a foot-pedal press stops it,
returns the arm and hand to that start, and a second press resumes the policy.
--traj-view draws the dataset's demonstrated trajectories with the live EEF pose.
No leader-to-follower retargeting is used.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime
import json
from dataclasses import replace
import math
from pathlib import Path
import select
import signal
import socket
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

CONTROL_ROOT = Path(__file__).resolve().parents[2]
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from inference import protocol
from inference.executor.async_predictor import AsyncPredictor
from data_collection.collection import config as defaults

MOTOR_IDS = tuple(defaults.TRIMORPH_MOTOR_IDS)
# Euclidean XYZ distance in meters, including observation -> first action.
MAX_EEF_JUMP_M = 0.15
# Independent per-action EMA for the arm pose and hand motor positions.
# filtered = alpha * target + (1 - alpha) * previous; 1.0 disables smoothing.
ARM_ACTION_EMA_ALPHA = 1.0 # action[:6]: XYZ + rotation vector
HAND_ACTION_EMA_ALPHA = 1.0  # action[6:]: 13 hand motor positions
# Multiply hand targets before EMA; arm targets always retain scale 1.0.
HAND_ACTION_SCALE = 1.03
# --rtc: the expected inference delay is the most actions executed while waiting
# for any of this many recent replies (the paper's conservative max).
RTC_DELAY_HISTORY = 10
# TriMi on the bench rig; the collection defaults are placeholders for the glove.
DEFAULT_HAND_API_REPO = CONTROL_ROOT.parent / "trimorph-hand-api"
DEFAULT_HAND_CONFIG = CONTROL_ROOT / "config" / "xl330_test_config.yaml"
# Every start the executor reaches, one JSON object per line; --start-id repeats one.
DEFAULT_START_LOG = CONTROL_ROOT / "output" / "executor_starts.jsonl"
# One folder per run: piper.csv (200 Hz feedback), commands.csv, events.csv, chunks.npz.
DEFAULT_RUN_LOG_ROOT = CONTROL_ROOT / "output" / "executor_runs"
PIPER_LOG_HZ = 200.0
EULER_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
PIPER_LOG_FIELDS = ("monotonic_ns", "joint_timestamp_s", *(f"joint_{i}" for i in range(1, 7)),
                    *(f"vel_{i}" for i in range(1, 7)), *(f"current_{i}" for i in range(1, 7)),
                    "fw_timestamp_s", *(f"fw_{name}" for name in EULER_FIELDS),
                    "arm_status", "motion_status")
JOINT_CONTROL_FIELDS = ("sigma", "damping", "step_scale", "pos_err_mm", "rot_err_deg")
COMMAND_LOG_FIELDS = ("monotonic_ns", "cycle", "index", *(f"raw_{name}" for name in EULER_FIELDS),
                      *EULER_FIELDS, *(f"cmd_joint_{i}" for i in range(1, 7)), *JOINT_CONTROL_FIELDS)
EVENT_LOG_FIELDS = ("monotonic_ns", "cycle", "event", "detail")
DEFAULT_TRAJ_DATASET = CONTROL_ROOT / "data_trimorph_hand_mocap_quest_4"
DEFAULT_HAND_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO9P0-if00-port0"


def vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return result


def pose_to_policy(pose):
    """Piper XYZ/Euler xyz radians -> policy XYZ/rotation-vector radians."""
    pose = vector(pose, 6, "Piper pose")
    return np.concatenate((pose[:3], Rotation.from_euler("xyz", pose[3:]).as_rotvec()))


def pose_to_piper(pose):
    """Policy XYZ/rotation-vector radians -> Piper XYZ/Euler xyz radians."""
    pose = vector(pose, 6, "policy pose")
    return np.concatenate((pose[:3], Rotation.from_rotvec(pose[3:]).as_euler("xyz")))


def jitter_start_joints(joints_deg, radius_m, angle_deg, rng, attempts=50):
    """Joints for a random flange pose within radius_m and angle_deg of the start's pose.

    The offset is uniform in a ball (position) and in a rotation-vector ball
    (orientation, base frame), solved by IK seeded at the start joints and checked
    with replay's joint margins, wrist margin, and Jacobian limit.
    Returns (joints_deg, offset_m, angle_deg).
    """
    joints_deg = vector(joints_deg, 6, "start joints")
    if radius_m <= 0 and angle_deg <= 0:
        return joints_deg, 0.0, 0.0
    from scipy.optimize import least_squares
    from data_collection.replay_actions import Limits, check_joints, piper_model

    fk, bounds = piper_model()
    limits = Limits()
    margin = np.radians(limits.joint_margin_deg)
    lower, upper = bounds[:, 0] + margin, bounds[:, 1] - margin
    seed = np.radians(joints_deg)
    start = fk(seed)
    start_rotation = Rotation.from_euler("xyz", start[3:])

    def ball(radius):
        direction = rng.normal(size=3)
        return direction / np.linalg.norm(direction) * radius * rng.uniform() ** (1 / 3)

    for _ in range(attempts):
        offset = ball(radius_m)
        turn = Rotation.from_rotvec(ball(np.radians(angle_deg)))
        position, rotation = start[:3] + offset, turn * start_rotation

        def residual(q):
            pose = fk(q)
            return np.r_[pose[:3] - position, .2 * (Rotation.from_euler("xyz", pose[3:]) * rotation.inv()).as_rotvec()]

        solution = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper), max_nfev=150,
                                 ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = residual(solution.x)
        if np.linalg.norm(error[:3]) > .0005 or np.linalg.norm(error[3:]) / .2 > np.radians(.2):
            continue
        try:
            check_joints(solution.x, bounds, fk, limits)
        except ValueError:
            continue
        return np.degrees(solution.x), float(np.linalg.norm(offset)), float(np.degrees(turn.magnitude()))
    raise RuntimeError(f"No valid IK for a random start offset in {attempts} attempts; reduce the start jitter")


def load_start_log(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_start_log(path, record):
    """Append a start record with the next id; return that id."""
    path = Path(path)
    record = {"id": len(load_start_log(path)), **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as log:
        log.write(json.dumps(record) + "\n")
    return record["id"]


def start_base(args):
    """(base joints deg, hand start name, label) chosen by --start-joints-deg, --start-id, or --start-home."""
    if args.start_joints_deg:
        return vector(args.start_joints_deg, 6, "start joints"), args.start_home, "custom"
    if args.start_id is not None:
        record = load_start_log(args.start_log)[args.start_id]
        return vector(record["joints_deg"], 6, "logged start joints"), record["hand_start"], f"start #{args.start_id}"
    return vector(defaults.PIPER_START_JOINTS_DEG[args.start_home], 6, "start joints"), args.start_home, args.start_home


def load_hand_class(api_repo):
    # The Trimorph fork still exports midas_hand_api.MidasHand. Ensure that an
    # unrelated MIDAS installation cannot silently supply the implementation.
    api_repo = Path(api_repo).expanduser().resolve()
    if not (api_repo / "midas_hand_api" / "__init__.py").is_file():
        raise FileNotFoundError(f"Trimorph hand API not found: {api_repo}")
    sys.path.insert(0, str(api_repo))
    import midas_hand_api
    if not Path(midas_hand_api.__file__).resolve().is_relative_to(api_repo):
        raise RuntimeError(f"Wrong hand API loaded: {midas_hand_api.__file__}")
    print(f"Hand API: {midas_hand_api.__file__}")
    return midas_hand_api.MidasHand


class RobotIO:
    def __init__(self, args):
        from data_collection.collection.hand_common import load_hand_config, verify_connected_hand
        from scripts.piper_api.piper_api import PiperAPI

        self.hand = None
        self.piper = None
        self.piper_log = None
        self.joint_control = None
        self.dt = 1.0 / args.hz
        self.commanded_pose = None
        self.segment = None
        # move_j commands per action (joint mode): 4 at --hz 15 and --command-hz 60.
        self.substeps = max(1, round(args.command_hz / args.hz)) if args.arm_control == "joint" else 1
        self.state_note = "measured"
        self.state_anchor = args.state_anchor
        self.anchor_max_mm, self.anchor_max_deg = args.anchor_max_mm, args.anchor_max_deg
        if args.arm_control == "joint":
            from inference.executor.joint_control import JointController
            self.joint_control = JointController(args.max_joint_speed_deg_s, sigma_start=args.ik_damping_sigma)
        try:
            hand_class = load_hand_class(args.hand_api_repo)
            config = load_hand_config(Path(args.hand_config).expanduser(), args.hand_port, MOTOR_IDS)
            # Match the existing Trimorph replay setup, not leader force feedback.
            config = replace(config, operating_mode=3)
            self.hand = hand_class(config=config, autoconnect=True)
            verify_connected_hand(self.hand, role="Trimorph executor")
            if tuple(self.hand.motor_ids) != MOTOR_IDS:
                raise RuntimeError("Hand motor order must be exactly 0..12")
            self.hand.configure(enable_torque=False)
            self.hand.set_motion_profile(
                profile_velocity_rad_s=args.hand_velocity,
                profile_acceleration_rad_s2=args.hand_acceleration,
            )
            self.hand.set_positions(vector(self.hand.read_pos(), 13, "hand state"), clip=True)
            self.hand.enable_torque()
            self.piper = PiperAPI(
                firmware=args.piper_firmware, interface=args.piper_interface,
                channel=args.piper_channel, installation_pos=args.piper_installation_pos,
                payload=args.piper_payload,
            )
            if not self.piper.connect() or not self.piper.enable():
                raise RuntimeError("Piper connection/enable failed")
            if not args.no_log:
                self.piper_log = PiperFeedbackLogger(self.piper)
            if not args.no_start_move:
                self.move_to_start(args)
            self.piper.set_speed_percent(args.piper_speed)
        except BaseException:
            self.close()
            raise

    def move_to_start(self, args, confirm=True):
        from scripts.piper_api.replay_trajectory import POSITION_MODE_SETTLE_S, read_joint_angles
        from scripts.teleop.quest_teleop import CONTROLLER_LIMITS_DEG, go_home, home_warnings

        joints, hand_name, name = start_base(args)
        for joint, (angle, (low, high)) in enumerate(zip(joints, CONTROLLER_LIMITS_DEG), start=1):
            if not low <= angle <= high:
                raise ValueError(f"Start J{joint} = {angle:g} deg is outside the controller limits {low}..{high}")
        # The jittered solution stays inside the SDK limits with replay's 3-degree margin.
        target, offset_m, angle_deg = jitter_start_joints(
            joints, args.start_jitter_mm / 1000, args.start_jitter_deg, np.random.default_rng())
        for warning in home_warnings(target):
            print(f"WARNING: start {warning}")
        current = np.degrees(read_joint_angles(self.piper))
        travel = np.max(np.abs(target - current))
        shown = name
        if offset_m or angle_deg:
            shown += f" + random {offset_m * 1000:.0f} mm, {angle_deg:.1f} deg"
        print(f"Start {shown} joints (deg): " + "  ".join(f"J{i+1} {v:7.2f}" for i, v in enumerate(target)))
        message = f"Largest joint travel {travel:.1f} deg at {args.start_speed_deg_s:g} deg/s."
        if confirm:
            input(f"{message} Clear the workspace, keep a hand near the e-stop, and press Enter to move "
                  "(Ctrl-C to abort)...")
        else:
            print(message, flush=True)
        self.piper.set_position_mode()
        time.sleep(POSITION_MODE_SETTLE_S)
        self.piper.set_speed_percent(100)  # The ramp in go_home sets the actual speed.
        go_home(self.piper, np.radians(target), args.start_speed_deg_s, 50.0)
        print("Reached start joints.", flush=True)
        if not args.no_hand_start:
            self.move_hand_to_start(hand_name)
        self.log_start(args, name, joints, target, offset_m, angle_deg, None if args.no_hand_start else hand_name)

    def log_start(self, args, name, base_deg, target_deg, offset_m, angle_deg, hand_name):
        from scripts.piper_api.replay_trajectory import read_joint_angles

        pose = self.piper.get_eef_pose()
        start_id = append_start_log(args.start_log, {
            "time": datetime.now().isoformat(timespec="seconds"),
            "base": name,
            "base_joints_deg": np.round(base_deg, 3).tolist(),
            "joints_deg": np.round(target_deg, 3).tolist(),
            "random_offset_mm": round(offset_m * 1000, 1),
            "random_angle_deg": round(angle_deg, 2),
            "measured_joints_deg": np.round(np.degrees(read_joint_angles(self.piper)), 3).tolist(),
            "flange_xyz_mm": None if pose is None else np.round(np.multiply(pose[:3], 1000), 1).tolist(),
            "flange_euler_xyz_deg": None if pose is None else np.round(np.degrees(pose[3:]), 2).tolist(),
            "hand_start": hand_name or args.start_home,
            "hand_moved": hand_name is not None,
        })
        print(f"Start #{start_id} logged in {args.start_log}; repeat it with --start-id {start_id} "
              f"(or --start-joints-deg {' '.join(f'{v:.2f}' for v in target_deg)}).", flush=True)

    def move_hand_to_start(self, name, tolerance_rad=0.05, timeout_s=5.0):
        # The motion profile set at startup limits the speed of this single target write.
        target = self.hand.clip_positions(vector(defaults.HAND_START_POSITIONS_RAD[name], 13, "hand start"))
        try:
            self.hand.set_positions_blocking(target, clip=False, tolerance_rad=tolerance_rad, timeout_s=timeout_s)
            print(f"Reached {name} hand start.", flush=True)
        except TimeoutError:
            error = np.abs(vector(self.hand.read_pos(), 13, "hand state") - target)
            off = ", ".join(f"motor {m} {e:.3f} rad" for m, e in enumerate(error) if e > tolerance_rad)
            print(f"WARNING: hand not within {tolerance_rad:g} rad of {name} start after {timeout_s:g} s: {off}",
                  flush=True)

    def return_to_start(self, args):
        """Pedal reset: no Enter prompt, then restore the policy speed."""
        self.move_to_start(args, confirm=False)
        self.piper.set_speed_percent(args.piper_speed)
        if self.joint_control is not None:
            self.joint_control.q = None  # restart joint targets from the measured start
        self.commanded_pose = None  # the next observation uses the measured start

    def eef_pose(self):
        """Measured flange pose (XYZ m + Euler xyz rad), or None if feedback is missing."""
        return self.piper.get_eef_pose()

    def read_state(self, anchor=None):
        """Observation state: arm pose (see anchored_pose) + measured hand positions."""
        pose = self.piper.get_eef_pose()
        if pose is None:
            raise RuntimeError("Piper EEF feedback unavailable")
        hand = vector(self.hand.read_pos(), 13, "hand state")
        return np.concatenate((pose_to_policy(self.anchored_pose(pose, anchor)), hand)).astype(np.float32)

    def anchored_pose(self, measured, anchor=None):
        """The pose the predictor anchors the next chunk to.

        The predictor starts each chunk exactly at the observed arm pose, so with
        the measured pose the arm's tracking lag or swing at observation time is
        copied into every chunk. --state-anchor command uses the last commanded
        pose instead (in joint mode, FK of the joint target sent), or `anchor`
        when given (--rtc: the raw policy action just commanded, so the new chunk
        continues the old one in its own frame), unless the arm is more than
        --anchor-max-mm / --anchor-max-deg away from it.
        """
        measured = vector(measured, 6, "measured pose")
        commanded = self.commanded_pose if anchor is None else vector(anchor, 6, "anchor pose")
        if self.state_anchor != "command" or commanded is None:
            self.state_note = "measured"
            return measured
        off_mm = float(np.linalg.norm(measured[:3] - commanded[:3]) * 1000)
        off_deg = float(np.degrees((Rotation.from_euler("xyz", measured[3:])
                                    * Rotation.from_euler("xyz", commanded[3:]).inv()).magnitude()))
        if off_mm <= self.anchor_max_mm and off_deg <= self.anchor_max_deg:
            source = "command" if anchor is None else "action"
            self.state_note = f"{source}; measured off {off_mm:.1f} mm {off_deg:.1f} deg"
            return commanded
        self.state_note = f"measured; FALLBACK, off {off_mm:.1f} mm {off_deg:.1f} deg"
        print(f"WARNING: arm is {off_mm:.0f} mm / {off_deg:.0f} deg from its commanded pose; "
              "sending the measured pose to the predictor", flush=True)
        return measured

    def apply_action(self, action):
        """Command the arm and hand; returns joint-control diagnostics (None with --arm-control pose)."""
        action = vector(action, 19, "action")
        info = None
        if self.joint_control is None:
            pose = pose_to_piper(action[:6])
            self.piper.move_eef(pose.tolist())
            self.commanded_pose = pose
        else:
            if self.joint_control.q is None:
                joints = self.piper.get_joint_angles()
                if joints is None:
                    raise RuntimeError("Piper joint feedback unavailable")
                self.joint_control.reset(joints)
            previous = self.joint_control.q.copy()
            target, info = self.joint_control.command(pose_to_piper(action[:6]), self.dt)
            # Reach the target over the action period in `substeps` evenly spaced move_j
            # commands (run() sends the rest) instead of one jump per action.
            self.segment = (previous, target)
            self.send_substep(1 / self.substeps)
            info["joints"] = target
            # Where the arm is actually being driven: reachable, after damping and speed caps.
            self.commanded_pose = np.asarray(self.joint_control.fk(target), dtype=float)
        # Positions already use leader coordinates. Signs/offsets are applied
        # exactly once by the hand API, with no follower scaling or retargeting.
        self.hand.set_positions(action[6:], clip=True)
        return info

    def send_substep(self, fraction):
        """Joint target `fraction` (0..1] of the way through the current action's segment."""
        start, end = self.segment
        self.piper.move_joint((start + fraction * (end - start)).tolist(), mode="smooth")

    def close(self):
        try:
            if self.hand is not None:
                self.hand.shutdown()
        finally:
            if self.piper_log is not None:
                self.piper_log.stop()
            if self.piper is not None:
                self.piper.disconnect()


class OakCamera:
    def __init__(self, args):
        from data_collection.collection.depthai_utils import (
            apply_camera_control, import_depthai, require_depthai_device,
        )
        from data_collection.collection.depthai_task_camera_worker import flip_frame
        import cv2

        self.cv2 = cv2
        self.flip_frame = flip_frame
        self.flip = args.cam_flip
        self.timeout = args.camera_timeout
        self.latest_frame = None
        self.window = "Wrist camera (policy input; q/Esc stops)" if args.show else None
        dai = import_depthai()
        require_depthai_device(dai)
        self.pipeline = dai.Pipeline()
        try:
            cam = self.pipeline.create(dai.node.Camera).build(sensorFps=args.cam_fps)
            # Same locked exposure/white balance as collection, so images match training.
            apply_camera_control(cam, exposure_us=args.cam_exposure_us, iso=args.cam_iso,
                                 white_balance_k=args.cam_white_balance_k)
            output = cam.requestOutput((args.cam_width, args.cam_height), dai.ImgFrame.Type.NV12,
                                       fps=args.cam_fps)
            self.queue = output.createOutputQueue()
            self.pipeline.start()
            # Match collection warmup, bounded by the configured timeout.
            self.read()
        except BaseException:
            self.close()
            raise

    def newest_bgr(self):
        """Newest queued frame, flipped like collection, or None if none is queued."""
        packet = self.queue.tryGet()
        if packet is None:
            return None
        # Prefer the newest available image.
        for _ in range(100):
            newer = self.queue.tryGet()
            if newer is None:
                break
            packet = newer
        bgr = packet.getCvFrame()
        if bgr is None:
            return None
        bgr = self.flip_frame(self.cv2, bgr, self.flip)
        self.latest_frame = (bgr, time.time(), time.monotonic())
        return bgr

    def read_available(self):
        """Nonblocking async observation, including frames consumed by preview()."""
        bgr = self.newest_bgr()
        if bgr is not None and self.window:
            self.show(bgr)
        if self.latest_frame is None:
            return None
        bgr, capture_time, received_at = self.latest_frame
        if time.monotonic() - received_at > self.timeout:
            raise TimeoutError("No OAK camera image within camera timeout")
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB), capture_time

    def show(self, bgr):
        self.cv2.imshow(self.window, bgr)
        if self.cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            raise KeyboardInterrupt

    def read(self):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            bgr = self.newest_bgr()
            if bgr is None:
                time.sleep(0.002)
                continue
            if self.window:
                self.show(bgr)
            return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        raise TimeoutError("No OAK camera image within camera timeout")

    def preview(self):
        """Refresh the window between actions; the policy only sees read() frames."""
        if self.window:
            bgr = self.newest_bgr()
            if bgr is not None:
                self.show(bgr)

    def close(self):
        if self.window:
            self.cv2.destroyAllWindows()
        self.pipeline.stop()


class PiperFeedbackLogger:
    """Poll the SDK's cached Piper feedback at PIPER_LOG_HZ on a background thread.

    Joint angles, motor velocity/current, firmware flange pose, and the
    controller's motion status (whether MOVE P reports its target reached) per row.
    """

    def __init__(self, api, rate_hz=PIPER_LOG_HZ):
        from scripts.piper_api.robot_helpers import _status_value

        self.api = api
        self.status_value = _status_value
        self.period_s = 1.0 / rate_hz
        self.rows = []
        self.errors = 0
        self.first_error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name="piper-feedback-log", daemon=True)
        self.thread.start()

    def run(self):
        next_t = time.monotonic()
        while not self.stop_event.is_set():
            # A failed read skips one sample; it must not end the log.
            try:
                self.sample()
            except Exception as exc:
                self.errors += 1
                self.first_error = self.first_error or repr(exc)
            next_t += self.period_s
            self.stop_event.wait(max(0.0, next_t - time.monotonic()))

    def sample(self):
        joints = self.api.get_joint_state()
        if joints is not None:
            row = {"monotonic_ns": time.monotonic_ns(), "joint_timestamp_s": f"{joints.timestamp:.6f}"}
            row.update({f"joint_{i}": f"{v:.9f}" for i, v in enumerate(joints.joints, start=1)})
            motors = self.api.get_motor_state()
            if motors is not None:
                row.update({f"vel_{i}": f"{v:.6f}" for i, v in enumerate(motors.velocities, start=1)})
                row.update({f"current_{i}": f"{v:.6f}" for i, v in enumerate(motors.currents, start=1)})
            eef = self.api.get_eef_state()
            if eef is not None:
                row["fw_timestamp_s"] = f"{eef.timestamp:.6f}"
                row.update({f"fw_{name}": f"{v:.9f}" for name, v in zip(EULER_FIELDS, eef.pose)})
            status = self.api.robot.get_arm_status()
            if status is not None:
                row["arm_status"] = self.status_value(status.msg.arm_status)
                row["motion_status"] = self.status_value(status.msg.motion_status)
            self.rows.append(row)

    def stop(self, timeout_s=1.0):
        """Stop polling; a read stuck in the SDK is reported, not waited on forever."""
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.thread.join(timeout_s)
        if self.thread.is_alive():
            print(f"WARNING: Piper feedback logger stuck after {len(self.rows)} rows (an SDK read did not return)",
                  flush=True)
        if self.errors:
            print(f"WARNING: {self.errors} Piper feedback samples failed; first: {self.first_error}", flush=True)


class RunLog:
    """Commands sent, raw predictor chunks, and loop events, saved with the Piper feedback."""

    def __init__(self, root):
        self.directory = Path(root) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.commands = []
        self.events = []
        self.chunks = []

    def event(self, cycle, event, detail=""):
        self.events.append({"monotonic_ns": time.monotonic_ns(), "cycle": cycle, "event": event, "detail": detail})

    def chunk(self, cycle, state, chunk):
        self.chunks.append((cycle, time.monotonic_ns(), np.asarray(state, np.float64), np.asarray(chunk, np.float64)))

    def command(self, cycle, index, raw, filtered, info=None):
        row = {"monotonic_ns": time.monotonic_ns(), "cycle": cycle, "index": index}
        if info is not None:  # --arm-control joint: the joint target actually sent, and IK diagnostics
            row.update({f"cmd_joint_{i}": f"{v:.9f}" for i, v in enumerate(info["joints"], start=1)})
            row.update({name: f"{info[name]:.6g}" for name in JOINT_CONTROL_FIELDS})
        row.update({f"raw_{name}": f"{v:.9f}" for name, v in zip(EULER_FIELDS, pose_to_piper(raw[:6]))})
        row.update({name: f"{v:.9f}" for name, v in zip(EULER_FIELDS, pose_to_piper(filtered[:6]))})
        self.commands.append(row)

    def save(self, args, piper_rows=None):
        if not self.commands and not self.events:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "meta.json").write_text(json.dumps(
            {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
             "arm_action_ema_alpha": ARM_ACTION_EMA_ALPHA, "piper_log_hz": PIPER_LOG_HZ}, indent=2) + "\n")
        for name, fields, rows in (("commands.csv", COMMAND_LOG_FIELDS, self.commands),
                                   ("events.csv", EVENT_LOG_FIELDS, self.events),
                                   ("piper.csv", PIPER_LOG_FIELDS, piper_rows)):
            if rows is None:
                continue
            with (self.directory / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        if self.chunks:
            # Ragged chunks are stored flat; chunk_cycle maps each row to its cycle.
            np.savez(self.directory / "chunks.npz",
                     cycle=np.array([c for c, *_ in self.chunks]),
                     received_monotonic_ns=np.array([t for _, t, *_ in self.chunks]),
                     state=np.stack([s for *_, s, _ in self.chunks]),
                     actions=np.concatenate([a for *_, a in self.chunks]),
                     chunk_cycle=np.concatenate([np.full(len(a), c) for c, *_, a in self.chunks]))
        print(f"Saved executor run log to {self.directory}", flush=True)

    def plot(self):
        if not (self.directory / "piper.csv").is_file() or not self.commands:
            return
        # Separate process: plotting must not touch the live --traj-view figure.
        import subprocess
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("plot_run.py")), str(self.directory)])
        if result.returncode:
            print(f"WARNING: plotting failed; rerun: python inference/executor/plot_run.py {self.directory}")


class FakeRobot:
    piper_log = None

    def __init__(self):
        self.state = np.zeros(19, dtype=np.float32)

    def read_state(self, anchor=None):
        state = self.state.copy()
        if anchor is not None:
            state[:6] = pose_to_policy(anchor)
        return state

    def apply_action(self, action):
        self.state = vector(action, 19, "fake action").astype(np.float32)

    def return_to_start(self, args):
        self.state = np.zeros(19, dtype=np.float32)

    def eef_pose(self):
        return pose_to_piper(self.state[:6])

    def close(self):
        pass


class FakeCamera:
    def read(self):
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def read_available(self):
        return self.read(), time.time()

    def preview(self):
        pass

    def close(self):
        pass


def pedal_pressed(pedal):
    return pedal is not None and bool(pedal.poll())


def refresh_views(robot, camera, view):
    camera.preview()
    if view is not None:
        view.update(robot.eef_pose())


def pause_at_start(args, robot, camera, pedal, view=None):
    """Return to the start pose, then hold until the next pedal press."""
    print("Pedal: stopping the policy and returning the arm and hand to the start.", flush=True)
    if view is not None:
        view.set_chunk(None)
    robot.return_to_start(args)
    pedal.poll()  # Discard presses made during the return.
    print("At start. Press the pedal to resume the policy (Ctrl-C to quit).", flush=True)
    while not pedal_pressed(pedal):
        refresh_views(robot, camera, view)
        time.sleep(0.02)
    print("Pedal: resuming the policy.", flush=True)


def ema_alphas():
    """Per-coordinate EMA weights for the policy actions: arm pose, then hand motors."""
    for name, alpha in (
        ("ARM_ACTION_EMA_ALPHA", ARM_ACTION_EMA_ALPHA),
        ("HAND_ACTION_EMA_ALPHA", HAND_ACTION_EMA_ALPHA),
    ):
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError(f"{name} must be finite and in (0, 1]")
    return np.r_[np.full(6, ARM_ACTION_EMA_ALPHA), np.full(13, HAND_ACTION_EMA_ALPHA)]


def filter_action(action, filtered, alpha):
    """(hand-scaled action, its EMA with the previous filtered action)."""
    action = np.array(action, dtype=np.float64, copy=True)
    action[6:] *= HAND_ACTION_SCALE
    return action, alpha * action + (1 - alpha) * filtered


def capture(robot, camera, seq, log=None, anchor=None):
    """Wrist image and observation state for request `seq`: (image, capture time, state, observe ms)."""
    if log is not None:
        log.event(seq, "observe")
    start = time.perf_counter()
    image = camera.read()
    capture_time = time.time()
    state = vector(robot.read_state() if anchor is None else robot.read_state(anchor), 19, "robot state")
    if log is not None:
        log.event(seq, "state", getattr(robot, "state_note", "measured"))
    return image, capture_time, state, (time.perf_counter() - start) * 1000


def receive_chunk(sock, seq, state, log=None):
    """The reply to request `seq` as (chunk, header); None, after a warning, if the chunk jumps."""
    reply = protocol.recv_action(sock)
    if reply is None:
        raise ConnectionError("Predictor disconnected")
    reply_seq, chunk, header = reply
    if reply_seq != seq:
        raise ValueError(f"Expected reply sequence {seq}, got {reply_seq}")
    chunk = np.asarray(chunk)
    if chunk.ndim != 2 or chunk.shape[1] != 19 or not len(chunk) or not np.isfinite(chunk).all():
        raise ValueError("Predictor action chunk must be finite, nonempty [T,19]")
    if log is not None:
        log.chunk(seq, state, chunk)
        log.event(seq, "chunk_received", f"infer_ms={header.get('infer_ms', 0):.1f} len={len(chunk)}")
    positions = np.vstack((state[:3], chunk[:, :3]))
    jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    violations = np.flatnonzero(jumps > MAX_EEF_JUMP_M)
    if len(violations):
        index = int(violations[0])
        source = "observation" if index == 0 else f"action {index - 1}"
        print(
            f"WARNING: refusing entire chunk cycle={seq}: EEF jump from {source} "
            f"to action {index} is {jumps[index] * 100:.2f} cm "
            f"> {MAX_EEF_JUMP_M * 100:g} cm. No actions from this chunk executed; "
            "executor stopping.",
            flush=True,
        )
        return None
    return chunk, header


def finish_action(robot, dt, first_sent):
    """The remaining interpolated joint targets toward the action whose first one went out at first_sent."""
    substeps = getattr(robot, "substeps", 1)
    for step in range(2, substeps + 1):
        remaining = first_sent + (step - 1) / substeps * dt - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        robot.send_substep(step / substeps)


def reply_ready(sock):
    """Whether the predictor's reply has started to arrive; never blocks."""
    return bool(select.select([sock], [], [], 0)[0])


def run(args, robot, camera, sock, pedal=None, view=None, log=None):
    action_alpha = ema_alphas()
    if getattr(args, "async_inference", False):
        return run_async(args, robot, camera, sock, action_alpha, pedal, view, log)
    dt = 1.0 / args.hz
    seq = 0
    filtered_action = None
    if pedal is not None:
        pedal.poll()  # Ignore presses made during setup and the start move.
    while not args.max_cycles or seq < args.max_cycles:
        if pedal_pressed(pedal):
            if log is not None:
                log.event(seq, "pedal_reset")
            pause_at_start(args, robot, camera, pedal, view)
            filtered_action = None
        image, capture_time, state, observe_ms = capture(robot, camera, seq, log)
        send_start = time.perf_counter()
        protocol.send_obs(sock, seq, image, state, t_capture=capture_time,
                          jpeg_quality=args.jpeg_quality)
        reply = receive_chunk(sock, seq, state, log)
        if reply is None:
            return
        chunk, header = reply
        execute_steps = planned_action_steps(args, chunk)
        rtt_ms = (time.perf_counter() - send_start) * 1000
        if view is not None:
            view.set_chunk(chunk[:execute_steps, :3])
        execute_start = time.perf_counter()
        overruns = 0
        if filtered_action is None:
            filtered_action = state.copy()
        interrupted = False
        for index, action in enumerate(chunk[:execute_steps]):
            if pedal_pressed(pedal):
                interrupted = True
                break
            action, filtered_action = filter_action(action, filtered_action, action_alpha)
            info = robot.apply_action(filtered_action)
            first_sent = time.perf_counter()  # after IK: later substeps keep even spacing from here
            if log is not None:
                log.command(seq, index, action, filtered_action, info)
            refresh_views(robot, camera, view)
            finish_action(robot, dt, first_sent)
            remaining = execute_start + (index + 1) * dt - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                overruns += 1
        print(f"cycle={seq} chunk={len(chunk)} hz={args.hz:g} observe={observe_ms:.1f}ms "
              f"infer={header.get('infer_ms', 0):.1f}ms rtt={rtt_ms:.1f}ms "
              f"execute={(time.perf_counter()-execute_start)*1000:.1f}ms overruns={overruns}"
              f"{f' interrupted after {index} actions' if interrupted else ''}", flush=True)
        if log is not None:
            log.event(seq, "chunk_done", f"overruns={overruns}{' interrupted' if interrupted else ''}")
        seq += 1
        if interrupted:
            if log is not None:
                log.event(seq, "pedal_reset")
            pause_at_start(args, robot, camera, pedal, view)
            filtered_action = None


def validate_chunk(chunk):
    chunk = np.asarray(chunk)
    if chunk.ndim != 2 or chunk.shape[1] != 19 or not len(chunk) or not np.isfinite(chunk).all():
        raise ValueError("Predictor action chunk must be finite, nonempty [T,19]")
    return chunk


def planned_action_steps(args, chunk):
    """Use a chosen prefix of the returned chunk, or all of it when uncapped."""
    limit = getattr(args, "execute_steps", 0)
    return min(limit, len(chunk)) if limit else len(chunk)


def check_chunk_jumps(seq, state, chunk, previous=None):
    positions = np.vstack((state[:3], chunk[:, :3]))
    jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    violations = np.flatnonzero(jumps > MAX_EEF_JUMP_M)
    if len(violations):
        index = int(violations[0])
        source = "observation" if index == 0 else f"action {index - 1}"
        distance = jumps[index]
    elif previous is not None and np.linalg.norm(chunk[0, :3] - previous[:3]) > MAX_EEF_JUMP_M:
        index, source = 0, "last commanded action"
        distance = np.linalg.norm(chunk[0, :3] - previous[:3])
    else:
        return True
    print(f"WARNING: refusing entire chunk cycle={seq}: EEF jump from {source} "
          f"to action {index} is {distance * 100:.2f} cm > {MAX_EEF_JUMP_M * 100:g} cm. "
          "No actions from this chunk executed; executor stopping.", flush=True)
    return False


def run_async(args, robot, camera, sock, action_alpha, pedal=None, view=None, log=None):
    """Overlap one prediction with playback; replace tails at policy action boundaries.

    Hardware reads/writes and GUI operations stay on this thread. JPEG encoding
    and the request/reply run on the socket worker. Complete an action's joint
    interpolation before switching chunks, without compressing the action clock.
    """
    dt = 1.0 / args.hz
    lead = args.inference_latency_ms / 1000
    predictor = AsyncPredictor(sock, args.jpeg_quality)
    seq = 0  # Number of requests submitted, including replies discarded on reset.
    chunk = None
    pending = None
    discard_pending = False
    waiting = False
    filtered_action = None
    request_at = time.perf_counter()
    camera_wait_since = request_at

    class PedalReset(Exception):
        pass

    def event(cycle, name, detail=""):
        if log is not None:
            log.event(cycle, name, detail)

    def can_request():
        return not args.max_cycles or seq < args.max_cycles

    def poll():
        nonlocal seq, pending, request_at, camera_wait_since
        if pedal_pressed(pedal):
            raise PedalReset
        predictor.check_error()
        now = time.perf_counter()
        if predictor.future is None and can_request() and now >= request_at:
            # No camera wait on the control thread. Preview caches its latest
            # frame too, so it cannot steal an observation from this request.
            observation = camera.read_available()
            if observation is None:
                if now - camera_wait_since > args.camera_timeout:
                    raise TimeoutError("No OAK camera image within camera timeout")
                return
            image, capture_time = observation
            event(seq, "observe")
            state = vector(robot.read_state(), 19, "robot state")
            event(seq, "state", getattr(robot, "state_note", "measured"))
            pending = (seq, state, (time.perf_counter() - now) * 1000)
            predictor.submit(seq, image, state, capture_time)
            seq += 1
            request_at = float("inf")

    def wait_until(deadline):
        while time.perf_counter() < deadline:
            poll()
            now = time.perf_counter()
            wake = min(deadline, now + 0.002)
            if predictor.future is None and can_request() and request_at > now:
                wake = min(wake, request_at)
            if wake > now:
                time.sleep(wake - now)

    def finish(reason):
        nonlocal chunk
        if chunk is None:
            return
        dropped = len(chunk) - index
        print(f"cycle={cycle} chunk={len(chunk)} planned={execute_steps} hz={args.hz:g} observe={observe_ms:.1f}ms "
              f"infer={header.get('infer_ms', 0):.1f}ms rtt={rtt_ms:.1f}ms "
              f"execute={(time.perf_counter() - execute_start) * 1000:.1f}ms "
              f"executed={index} dropped={dropped} overruns={overruns} {reason}", flush=True)
        event(cycle, "chunk_done",
              f"planned={execute_steps} executed={index} dropped={dropped} overruns={overruns} {reason}")
        chunk = None

    if pedal is not None:
        pedal.poll()
    try:
        while True:
            try:
                poll()
                # This is always an action boundary: the previous action and all
                # of its interpolated commands have finished before taking a reply.
                if predictor.ready():
                    reply, new_rtt_ms = predictor.take()
                    new_cycle, state, new_observe_ms = pending
                    pending = None
                    if discard_pending:
                        event(new_cycle, "chunk_discarded", "pedal reset")
                        discard_pending = False
                        request_at = camera_wait_since = time.perf_counter()
                        continue
                    new_chunk = validate_chunk(reply[1])
                    if log is not None:
                        log.chunk(new_cycle, state, new_chunk)
                    event(new_cycle, "chunk_received",
                          f"infer_ms={reply[2].get('infer_ms', 0):.1f} len={len(new_chunk)} rtt_ms={new_rtt_ms:.1f}")
                    if not check_chunk_jumps(new_cycle, state, new_chunk, filtered_action):
                        finish("rejected replacement")
                        return
                    finish("replaced")
                    if waiting:
                        event(new_cycle, "wait_end")
                        waiting = False
                    cycle, chunk, index = new_cycle, new_chunk, 0
                    execute_steps = planned_action_steps(args, chunk)
                    header, rtt_ms, observe_ms = reply[2], new_rtt_ms, new_observe_ms
                    execute_start, overruns = time.perf_counter(), 0
                    if filtered_action is None:
                        filtered_action = state.copy()
                    if view is not None:
                        view.set_chunk(chunk[:execute_steps, :3])
                    event(cycle, "chunk_start", f"planned={execute_steps} predicted={len(chunk)}")

                if chunk is None:
                    if predictor.future is None and not can_request():
                        return
                    if not waiting:
                        event(seq, "wait_start")
                        waiting = True
                    refresh_views(robot, camera, view)
                    time.sleep(0.002)
                    continue

                action, filtered_action = filter_action(chunk[index], filtered_action, action_alpha)
                info = robot.apply_action(filtered_action)
                first_sent = time.perf_counter()
                if log is not None:
                    log.command(cycle, index, action, filtered_action, info)
                index += 1
                # Recompute from actual send time: slow IK/IO shifts the forecast,
                # and a late reply never causes a burst of catch-up commands.
                next_action_at = first_sent + dt
                if predictor.future is None and can_request():
                    request_at = next_action_at + (execute_steps - index) * dt - lead
                    camera_wait_since = max(first_sent, request_at)
                refresh_views(robot, camera, view)
                substeps = getattr(robot, "substeps", 1)
                for step in range(2, substeps + 1):
                    wait_until(first_sent + (step - 1) / substeps * dt)
                    robot.send_substep(step / substeps)
                if time.perf_counter() > next_action_at:
                    overruns += 1
                wait_until(next_action_at)
                if index == execute_steps:
                    finish("complete")
            except PedalReset:
                finish("pedal reset")
                if waiting:
                    event(seq, "wait_end")
                    waiting = False
                event(seq, "pedal_reset")
                # Keep the sole socket request until its reply is drained, but
                # never execute a prediction made before returning to the start.
                discard_pending = predictor.future is not None
                pause_at_start(args, robot, camera, pedal, view)
                filtered_action = None
                request_at = camera_wait_since = time.perf_counter()
    finally:
        predictor.close()


def check_rtc_chunk(args, chunk, length=None):
    if len(chunk) < args.rtc_horizon + 2:
        raise ValueError(f"--rtc-horizon {args.rtc_horizon} needs chunks of at least {args.rtc_horizon + 2} "
                         f"actions; the predictor sent {len(chunk)}")
    if length is not None and len(chunk) != length:
        raise ValueError(f"RTC chunk length changed from {length} to {len(chunk)}")


def run_rtc(args, robot, camera, sock, pedal=None, view=None, log=None):
    """Real-time chunking: the arm does not wait for the predictor (inference/predictor/rtc.py).

    Chunk index 0 is the observation (the policy's actions are relative to the
    observed state), so every chunk is executed from index 1, one action per
    tick at --hz. Once --rtc-horizon of its actions have run (counting those run
    from the previous chunk while this one was predicted), the next request goes
    out right after action k0 is commanded: the observed arm pose is that raw
    action (anchored_pose), and the request carries chunk[k0:] and the expected
    delay d, the most actions run while waiting for any of the last
    RTC_DELAY_HISTORY replies. If the arm is too far from that action, the
    measured pose goes out with no chunk: the policy replans from the arm
    instead of continuing the plan. The arm keeps executing the old chunk; the new
    one, whose index i continues old step k0 + i, takes over at the first tick
    after its reply arrives: at index m + 1 after m more old actions. If the old
    chunk runs out first, the arm holds its last target. Only the first chunk
    after the start or a pedal reset is waited for, with the arm at rest.
    """
    action_alpha = ema_alphas()
    dt = 1.0 / args.hz
    seq = 0
    if pedal is not None:
        pedal.poll()  # Ignore presses made during setup and the start move.
    while not args.max_cycles or seq < args.max_cycles:
        image, capture_time, state, observe_ms = capture(robot, camera, seq, log)
        send_start = time.perf_counter()
        protocol.send_obs(sock, seq, image, state, t_capture=capture_time, jpeg_quality=args.jpeg_quality,
                          rtc={"prev_chunk": np.zeros((0, 19)), "delay": 0, "max_guidance": args.rtc_max_guidance})
        reply = receive_chunk(sock, seq, state, log)
        if reply is None:
            return
        chunk, header = reply
        rtt = time.perf_counter() - send_start
        check_rtc_chunk(args, chunk)
        print(f"cycle={seq} rtc first chunk={len(chunk)} hz={args.hz:g} observe={observe_ms:.1f}ms "
              f"infer={header.get('infer_ms', 0):.1f}ms rtt={rtt * 1000:.1f}ms", flush=True)
        delays = deque([math.ceil(rtt / dt)], maxlen=RTC_DELAY_HISTORY)
        chunk_seq, index, seq = seq, 1, seq + 1
        filtered_action = state.copy()
        pending = None
        overruns = starved = 0
        interrupted = False
        if view is not None:
            view.set_chunk(chunk[:, :3])
        next_tick = time.perf_counter()
        while True:
            if pedal_pressed(pedal):
                interrupted = True
                break
            if pending is not None and reply_ready(sock):
                m, k0 = pending["executed"], pending["k0"]
                if log is not None:
                    log.event(chunk_seq, "chunk_done", f"overruns={overruns} starved={starved}")
                reply = receive_chunk(sock, pending["seq"], pending["state"], log)
                if reply is None:
                    return
                new, header = reply
                check_rtc_chunk(args, new, len(chunk))
                delays.append(m)
                # The last old action run is chunk[k0 + m]; new[m + 1] follows it.
                step_mm = np.linalg.norm(new[m + 1, :3] - chunk[k0 + m, :3]) * 1000
                old_step_mm = (np.linalg.norm(chunk[k0 + m + 1, :3] - chunk[k0 + m, :3]) * 1000
                               if k0 + m + 1 < len(chunk) else float("nan"))
                ran = np.arange(1, m + 1)  # new steps whose old counterparts ran during inference
                ran_mm = np.linalg.norm(new[ran, :3] - chunk[k0 + ran, :3], axis=1).max() * 1000 if m else 0.0
                print(f"cycle={pending['seq']} rtc delay={m} (expected {pending['delay']}) "
                      f"infer={header.get('infer_ms', 0):.1f}ms "
                      f"wait={(time.perf_counter() - pending['sent']) * 1000:.0f}ms "
                      f"takeover step={step_mm:.1f}mm (old plan {old_step_mm:.1f}mm) "
                      f"off executed={ran_mm:.1f}mm overruns={overruns} starved={starved}", flush=True)
                if log is not None:
                    log.event(pending["seq"], "rtc_takeover",
                              f"delay={m} expected={pending['delay']} step_mm={step_mm:.2f} "
                              f"old_step_mm={old_step_mm:.2f} off_executed_mm={ran_mm:.2f}")
                chunk, chunk_seq, index = new, pending["seq"], m + 1
                pending = None
                overruns = starved = 0
                if view is not None:
                    view.set_chunk(chunk[:, :3])
            first_sent = None
            if index < len(chunk):
                action, filtered_action = filter_action(chunk[index], filtered_action, action_alpha)
                info = robot.apply_action(filtered_action)
                first_sent = time.perf_counter()  # after IK: later substeps keep even spacing from here
                if log is not None:
                    log.command(chunk_seq, index, action, filtered_action, info)
                index += 1
                if pending is not None:
                    pending["executed"] += 1
            elif pending is None:
                break  # --max-cycles reached and its last chunk executed
            else:
                starved += 1  # the old chunk ran out first: hold the last target
            if (pending is None and index - 1 >= args.rtc_horizon
                    and (not args.max_cycles or seq < args.max_cycles)):
                k0 = index - 1  # the action just commanded
                image, capture_time, state, _ = capture(robot, camera, seq, log, anchor=pose_to_piper(chunk[k0, :6]))
                # Measured fallback (the arm is far from the plan): replan from the arm, do not continue the plan.
                prev = chunk[k0:] if np.allclose(state[:3], chunk[k0, :3], rtol=0, atol=1e-6) else chunk[:0]
                delay = max(delays)
                protocol.send_obs(sock, seq, image, state, t_capture=capture_time, jpeg_quality=args.jpeg_quality,
                                  rtc={"prev_chunk": prev, "delay": delay, "max_guidance": args.rtc_max_guidance})
                pending = {"seq": seq, "k0": k0, "state": state, "delay": delay, "executed": 0,
                           "sent": time.perf_counter()}
                seq += 1
            refresh_views(robot, camera, view)
            if first_sent is not None:
                finish_action(robot, dt, first_sent)
            next_tick += dt
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                overruns += 1
                next_tick = time.perf_counter()  # late: continue from now instead of rushing to catch up
        if log is not None:
            log.event(chunk_seq, "chunk_done",
                      f"overruns={overruns} starved={starved}{' interrupted' if interrupted else ''}")
        if not interrupted:
            return
        if pending is not None:
            protocol.recv_action(sock)  # its chunk continues motion the reset abandoned
        if log is not None:
            log.event(seq, "pedal_reset")
        pause_at_start(args, robot, camera, pedal, view)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictor", required=True, help="GPU predictor hostname/IP")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--hz", type=float, default=15.0, help="Match camera sync rate / training stride")
    p.add_argument("--execute-steps", type=int, default=0,
                   help="Maximum actions to execute from each returned chunk; 0 uses the whole chunk (default). "
                        "Async timing uses this count, capped to the returned length")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--async-inference", action="store_true",
                   help="Predict during playback and replace remaining actions when the reply is ready")
    p.add_argument("--inference-latency-ms", type=float, default=70.0,
                   help="Async request lead time before the estimated chunk end (default: %(default)s ms)")
    p.add_argument("--max-cycles", type=int, default=0)
    p.add_argument("--timeout", type=float, default=30.0, help="TCP timeout in seconds")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--fake", action="store_true", help="Synthetic observations; no hardware imports/connections")
    p.add_argument("--hand-api-repo", type=Path, default=DEFAULT_HAND_API_REPO)
    p.add_argument("--hand-config", type=Path, default=DEFAULT_HAND_CONFIG)
    p.add_argument("--hand-port", default=DEFAULT_HAND_PORT)
    p.add_argument("--hand-velocity", type=float, default=defaults.FOLLOWER_PROFILE_VELOCITY_RAD_S)
    p.add_argument("--hand-acceleration", type=float, default=defaults.FOLLOWER_PROFILE_ACCELERATION_RAD_S2)
    for name, default in (
        ("interface", defaults.DEFAULT_PIPER_INTERFACE), ("channel", defaults.DEFAULT_PIPER_CHANNEL),
        ("firmware", defaults.DEFAULT_PIPER_FIRMWARE), ("installation-pos", defaults.DEFAULT_PIPER_INSTALLATION_POS),
        ("payload", defaults.DEFAULT_PIPER_PAYLOAD),
    ):
        p.add_argument(f"--piper-{name}", default=default)
    p.add_argument("--piper-speed", type=int, default=100)
    p.add_argument("--arm-control", choices=("joint", "pose"), default="joint",
                   help="joint: local damped IK with joint speed caps, sent with move_j (default); "
                        "pose: firmware Cartesian move_p (previous behavior)")
    p.add_argument("--max-joint-speed-deg-s", type=float, nargs=6, metavar="DEG_S",
                   default=None, help="Joint speed caps J1..J6 for --arm-control joint "
                                      "(default: inference/executor/joint_control.py)")
    p.add_argument("--command-hz", type=float, default=60.0,
                   help="With --arm-control joint, send interpolated move_j targets at about this rate "
                        "(rounded to a whole number per action: 4 per action at --hz 15); "
                        "set it to --hz for one jump per action")
    modes.add_argument("--rtc", action="store_true",
                   help="Real-time chunking: request the next chunk while executing the current one, guided to "
                        "continue it, so the arm does not wait for the predictor (needs this repo's predictor)")
    p.add_argument("--rtc-horizon", type=int, default=8,
                   help="With --rtc, actions executed from each chunk (including those run while it was "
                        "predicted) before the next request")
    p.add_argument("--rtc-max-guidance", type=float, default=20.0,
                   help="With --rtc, clip of the guidance toward the previous chunk (the RTC paper's beta, "
                        "inference/predictor/rtc.py); 0 disables guidance")
    p.add_argument("--state-anchor", choices=("command", "measured"), default="command",
                   help="Arm pose sent as the observation state, which the predictor anchors each chunk to: "
                        "the last commanded pose (default) or the measured pose (previous behavior)")
    p.add_argument("--anchor-max-mm", type=float, default=50.0,
                   help="With --state-anchor command, send the measured pose instead when the arm is "
                        "farther than this from the commanded pose")
    p.add_argument("--anchor-max-deg", type=float, default=20.0,
                   help="Same fallback for orientation")
    p.add_argument("--ik-damping-sigma", type=float, default=None,
                   help="IK damping starts when the Jacobian's smallest singular value is below this "
                        "(default: inference/executor/joint_control.py; 0 = exact IK)")
    p.add_argument("--start-home", choices=sorted(defaults.PIPER_START_JOINTS_DEG),
                   default=defaults.DEFAULT_EXECUTOR_START,
                   help="Named joint-space start before the first observation (default: %(default)s)")
    p.add_argument("--start-joints-deg", type=float, nargs=6, metavar="DEG", default=None,
                   help="Explicit J1..J6 start in degrees; overrides --start-home")
    p.add_argument("--start-speed-deg-s", type=float, default=10.0,
                   help="Speed of the fastest-moving joint during the start move")
    p.add_argument("--start-id", type=int, default=None,
                   help="Repeat a start logged in --start-log (its joints and hand start)")
    p.add_argument("--start-log", type=Path, default=DEFAULT_START_LOG,
                   help="Every start reached is appended here (default: %(default)s)")
    p.add_argument("--start-jitter-mm", type=float, default=0.0,
                   help="Random flange position offset (up to this radius) around the start; logged; default off")
    p.add_argument("--start-jitter-deg", type=float, default=0.0,
                   help="Random flange orientation offset (up to this angle) around the start; logged; default off")
    p.add_argument("--no-start-move", action="store_true", help="Start from wherever the arm and hand are")
    p.add_argument("--no-hand-start", action="store_true",
                   help="Move only the arm; the hand holds its position (hand start always follows --start-home)")
    p.add_argument("--cam-width", type=int, default=defaults.TASK_CAMERA_WIDTH)
    p.add_argument("--cam-height", type=int, default=defaults.TASK_CAMERA_HEIGHT)
    p.add_argument("--cam-flip", default=defaults.TASK_CAMERA_FLIP,
                   choices=["none", "horizontal", "vertical", "both", "180"])
    p.add_argument("--cam-fps", type=int, default=defaults.TASK_CAMERA_FPS)
    p.add_argument("--cam-exposure-us", type=int, default=defaults.TASK_CAMERA_EXPOSURE_US)
    p.add_argument("--cam-iso", type=int, default=defaults.TASK_CAMERA_ISO)
    p.add_argument("--cam-white-balance-k", type=int, default=defaults.TASK_CAMERA_WHITE_BALANCE_K)
    p.add_argument("--camera-timeout", type=float, default=5.0)
    p.add_argument("--no-pedal", action="store_true",
                   help="Run without the foot pedal (no pedal return-to-start); implied by --fake")
    p.add_argument("--foot-pedal-device", help="Foot pedal input device (default: auto-detect)")
    p.add_argument("--traj-view", nargs="?", type=Path, const=DEFAULT_TRAJ_DATASET, default=None,
                   metavar="DATASET",
                   help="Show demonstrated trajectories and the live EEF pose in 3D "
                        f"(default dataset: {DEFAULT_TRAJ_DATASET.name})")
    p.add_argument("--show", action="store_true",
                   help="Show the wrist camera live while executing; q/Esc stops like Ctrl-C")
    p.add_argument("--log-root", type=Path, default=DEFAULT_RUN_LOG_ROOT,
                   help="Save each run's Piper feedback (200 Hz), commands, and chunks under here "
                        "(default: %(default)s)")
    p.add_argument("--no-log", action="store_true", help="Do not save the run log")
    args = p.parse_args(argv)
    if args.execute_steps < 0:
        p.error("--execute-steps must be nonnegative (0 uses the whole chunk)")
    if not np.isfinite(args.inference_latency_ms) or args.inference_latency_ms < 0:
        p.error("--inference-latency-ms must be finite and nonnegative")
    if args.rtc and args.execute_steps:
        p.error("--execute-steps applies to synchronous/async playback; use --rtc-horizon with --rtc")
    for name in ("hz", "timeout", "camera_timeout", "hand_velocity", "hand_acceleration", "start_speed_deg_s"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    from inference.executor.joint_control import DAMPING_SIGMA, DEFAULT_MAX_JOINT_SPEED_DEG_S
    if args.max_joint_speed_deg_s is None:
        args.max_joint_speed_deg_s = list(DEFAULT_MAX_JOINT_SPEED_DEG_S)
    if args.ik_damping_sigma is None:
        args.ik_damping_sigma = DAMPING_SIGMA
    if not all(0 < v <= 180 for v in args.max_joint_speed_deg_s) or not 0 <= args.ik_damping_sigma <= 0.2:
        p.error("Require --max-joint-speed-deg-s in (0, 180] and --ik-damping-sigma in [0, 0.2]")
    if not args.hz <= args.command_hz <= 200:
        p.error("Require --hz <= --command-hz <= 200")
    if args.rtc_horizon < 1 or not 0 <= args.rtc_max_guidance <= 1000:
        p.error("Require --rtc-horizon >= 1 and --rtc-max-guidance in [0, 1000]")
    if args.rtc and args.state_anchor != "command":
        p.error("--rtc continues the commanded plan from the action just sent; use --state-anchor command")
    if not args.anchor_max_mm > 0 or not args.anchor_max_deg > 0:
        p.error("Require positive --anchor-max-mm and --anchor-max-deg")
    if args.start_speed_deg_s > 30:
        p.error("--start-speed-deg-s above 30 is not allowed")
    if not 0 <= args.start_jitter_mm <= 100 or not 0 <= args.start_jitter_deg <= 30:
        p.error("Require --start-jitter-mm 0..100 and --start-jitter-deg 0..30")
    if args.start_id is not None:
        if args.start_joints_deg:
            p.error("Use either --start-id or --start-joints-deg")
        logged = load_start_log(args.start_log)
        if not 0 <= args.start_id < len(logged):
            p.error(f"--start-id {args.start_id} is not in {args.start_log} ({len(logged)} starts logged)")
    if args.max_cycles < 0 or not 1 <= args.jpeg_quality <= 100 or not 1 <= args.piper_speed <= 100:
        p.error("Require max-cycles >= 0, jpeg-quality 1..100, piper-speed 1..100")
    if args.cam_fps <= 0 or not 0 < args.cam_exposure_us <= 1_000_000 / args.cam_fps:
        p.error("Require cam-fps > 0 and cam-exposure-us within one frame")
    if not 100 <= args.cam_iso <= 1600 or not 1000 <= args.cam_white_balance_k <= 12000:
        p.error("Require cam-iso 100..1600 and cam-white-balance-k 1000..12000")
    return args


def main():
    args = parse_args()
    robot = camera = sock = pedal = view = None
    log = None if args.no_log else RunLog(args.log_root)
    try:
        # Establish predictor connectivity before enabling any actuators.
        sock = socket.create_connection((args.predictor, args.port), timeout=args.timeout)
        protocol.set_low_latency(sock)
        if not args.no_pedal and not args.fake:
            from scripts.piper_api.foot_pedal import FootPedalTrigger
            pedal = FootPedalTrigger(args.foot_pedal_device, debounce_s=0.3)
            pedal.start()
            print(f"Foot pedal: {pedal.active_device}. Press it during the policy to return to the start.")
        if args.traj_view is not None:
            from inference.executor.trajectory_view import TrajectoryView
            view = TrajectoryView(args.traj_view)
            view.update()
        camera = FakeCamera() if args.fake else OakCamera(args)
        robot = FakeRobot() if args.fake else RobotIO(args)
        (run_rtc if args.rtc else run)(args, robot, camera, sock, pedal, view, log)
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        # Save before closing hardware, with Ctrl-C ignored: a second Ctrl-C or a
        # stuck close must not lose the log.
        if log is not None:
            piper_log = getattr(robot, "piper_log", None)
            if piper_log is not None:
                piper_log.stop()
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                log.save(args, None if piper_log is None else list(piper_log.rows))
            except Exception as exc:
                print(f"WARNING: saving the run log failed: {exc!r}", flush=True)
            finally:
                signal.signal(signal.SIGINT, previous)
        try:
            if robot is not None:
                robot.close()
        finally:
            try:
                if camera is not None:
                    camera.close()
            finally:
                try:
                    if sock is not None:
                        sock.close()
                finally:
                    if pedal is not None:
                        pedal.stop()
                    if view is not None:
                        view.close()
                    if log is not None:
                        log.plot()


if __name__ == "__main__":
    main()
