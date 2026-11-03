"""Execute 19D policy chunks on Trimorph + Piper, using collection coordinates.

The request/reply loop is synchronous: prediction latency remains a pause
between chunks. Timings explicitly expose that cost; this is not an async
controller. Before the first observation the arm ramps in joint space to the
go_to_joints.py demo start (or --start-joints-deg); --no-start-move skips it.
No leader-to-follower retargeting is used.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import socket
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

CONTROL_ROOT = Path(__file__).resolve().parents[2]
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from inference import protocol
from data_collection.collection import config as defaults

MOTOR_IDS = tuple(defaults.TRIMORPH_MOTOR_IDS)
# Euclidean XYZ distance in meters, including observation -> first action.
MAX_EEF_JUMP_M = 0.15
# TriMi on the bench rig; the collection defaults are placeholders for the glove.
DEFAULT_HAND_API_REPO = CONTROL_ROOT.parent / "trimorph-hand-api"
DEFAULT_HAND_CONFIG = CONTROL_ROOT / "config" / "xl330_test_config.yaml"
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
            if not args.no_start_move:
                self.move_to_start(args)
            self.piper.set_speed_percent(args.piper_speed)
        except BaseException:
            self.close()
            raise

    def move_to_start(self, args):
        from scripts.piper_api.go_to_joints import DEMO_START_JOINTS_DEG
        from scripts.piper_api.replay_trajectory import POSITION_MODE_SETTLE_S, read_joint_angles
        from scripts.teleop.quest_teleop import CONTROLLER_LIMITS_DEG, go_home, home_warnings

        target = vector(args.start_joints_deg or DEMO_START_JOINTS_DEG, 6, "start joints")
        for joint, (angle, (low, high)) in enumerate(zip(target, CONTROLLER_LIMITS_DEG), start=1):
            if not low <= angle <= high:
                raise ValueError(f"Start J{joint} = {angle:g} deg is outside the controller limits {low}..{high}")
        for warning in home_warnings(target):
            print(f"WARNING: start {warning}")
        current = np.degrees(read_joint_angles(self.piper))
        travel = np.max(np.abs(target - current))
        print("Start joints (deg): " + "  ".join(f"J{i+1} {v:7.2f}" for i, v in enumerate(target)))
        input(f"Largest joint travel {travel:.1f} deg at {args.start_speed_deg_s:g} deg/s. Clear the "
              "workspace, keep a hand near the e-stop, and press Enter to move (Ctrl-C to abort)...")
        self.piper.set_position_mode()
        time.sleep(POSITION_MODE_SETTLE_S)
        self.piper.set_speed_percent(100)  # The ramp in go_home sets the actual speed.
        go_home(self.piper, np.radians(target), args.start_speed_deg_s, 50.0)
        print("Reached start joints.", flush=True)

    def read_state(self):
        pose = self.piper.get_eef_pose()
        if pose is None:
            raise RuntimeError("Piper EEF feedback unavailable")
        hand = vector(self.hand.read_pos(), 13, "hand state")
        return np.concatenate((pose_to_policy(pose), hand)).astype(np.float32)

    def apply_action(self, action):
        action = vector(action, 19, "action")
        self.piper.move_eef(pose_to_piper(action[:6]).tolist())
        # Positions already use leader coordinates. Signs/offsets are applied
        # exactly once by the hand API, with no follower scaling or retargeting.
        self.hand.set_positions(action[6:], clip=True)

    def close(self):
        try:
            if self.hand is not None:
                self.hand.shutdown()
        finally:
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
        # Prefer the newest available image after executing a whole chunk.
        for _ in range(100):
            newer = self.queue.tryGet()
            if newer is None:
                break
            packet = newer
        bgr = packet.getCvFrame()
        return None if bgr is None else self.flip_frame(self.cv2, bgr, self.flip)

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


class FakeRobot:
    def __init__(self):
        self.state = np.zeros(19, dtype=np.float32)

    def read_state(self):
        return self.state.copy()

    def apply_action(self, action):
        self.state = vector(action, 19, "fake action").astype(np.float32)

    def close(self):
        pass


class FakeCamera:
    def read(self):
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def preview(self):
        pass

    def close(self):
        pass


def run(args, robot, camera, sock):
    dt = 1.0 / args.hz
    seq = 0
    while not args.max_cycles or seq < args.max_cycles:
        cycle_start = time.perf_counter()
        image = camera.read()
        capture_time = time.time()
        state = vector(robot.read_state(), 19, "robot state")
        observe_ms = (time.perf_counter() - cycle_start) * 1000
        send_start = time.perf_counter()
        protocol.send_obs(sock, seq, image, state, t_capture=capture_time,
                          jpeg_quality=args.jpeg_quality)
        reply = protocol.recv_action(sock)
        if reply is None:
            raise ConnectionError("Predictor disconnected")
        reply_seq, chunk, header = reply
        if reply_seq != seq:
            raise ValueError(f"Expected reply sequence {seq}, got {reply_seq}")
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or chunk.shape[1] != 19 or not len(chunk) or not np.isfinite(chunk).all():
            raise ValueError("Predictor action chunk must be finite, nonempty [T,19]")
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
            return
        rtt_ms = (time.perf_counter() - send_start) * 1000
        execute_start = time.perf_counter()
        overruns = 0
        for index, action in enumerate(chunk):
            robot.apply_action(action)
            camera.preview()
            remaining = execute_start + (index + 1) * dt - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                overruns += 1
        print(f"cycle={seq} chunk={len(chunk)} hz={args.hz:g} observe={observe_ms:.1f}ms "
              f"infer={header.get('infer_ms', 0):.1f}ms rtt={rtt_ms:.1f}ms "
              f"execute={(time.perf_counter()-execute_start)*1000:.1f}ms overruns={overruns}", flush=True)
        seq += 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictor", required=True, help="GPU predictor hostname/IP")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--hz", type=float, default=15.0, help="Match camera sync rate / training stride")
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
    p.add_argument("--start-joints-deg", type=float, nargs=6, metavar="DEG", default=None,
                   help="Joint-space start before the first observation "
                        "(default: DEMO_START_JOINTS_DEG in scripts/piper_api/go_to_joints.py)")
    p.add_argument("--start-speed-deg-s", type=float, default=10.0,
                   help="Speed of the fastest-moving joint during the start move")
    p.add_argument("--no-start-move", action="store_true", help="Start from wherever the arm is")
    p.add_argument("--cam-width", type=int, default=defaults.TASK_CAMERA_WIDTH)
    p.add_argument("--cam-height", type=int, default=defaults.TASK_CAMERA_HEIGHT)
    p.add_argument("--cam-flip", default=defaults.TASK_CAMERA_FLIP,
                   choices=["none", "horizontal", "vertical", "both", "180"])
    p.add_argument("--cam-fps", type=int, default=defaults.TASK_CAMERA_FPS)
    p.add_argument("--cam-exposure-us", type=int, default=defaults.TASK_CAMERA_EXPOSURE_US)
    p.add_argument("--cam-iso", type=int, default=defaults.TASK_CAMERA_ISO)
    p.add_argument("--cam-white-balance-k", type=int, default=defaults.TASK_CAMERA_WHITE_BALANCE_K)
    p.add_argument("--camera-timeout", type=float, default=5.0)
    p.add_argument("--show", action="store_true",
                   help="Show the wrist camera live while executing; q/Esc stops like Ctrl-C")
    args = p.parse_args(argv)
    for name in ("hz", "timeout", "camera_timeout", "hand_velocity", "hand_acceleration", "start_speed_deg_s"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.start_speed_deg_s > 30:
        p.error("--start-speed-deg-s above 30 is not allowed")
    if args.max_cycles < 0 or not 1 <= args.jpeg_quality <= 100 or not 1 <= args.piper_speed <= 100:
        p.error("Require max-cycles >= 0, jpeg-quality 1..100, piper-speed 1..100")
    if args.cam_fps <= 0 or not 0 < args.cam_exposure_us <= 1_000_000 / args.cam_fps:
        p.error("Require cam-fps > 0 and cam-exposure-us within one frame")
    if not 100 <= args.cam_iso <= 1600 or not 1000 <= args.cam_white_balance_k <= 12000:
        p.error("Require cam-iso 100..1600 and cam-white-balance-k 1000..12000")
    return args


def main():
    args = parse_args()
    robot = camera = sock = None
    try:
        # Establish predictor connectivity before enabling any actuators.
        sock = socket.create_connection((args.predictor, args.port), timeout=args.timeout)
        protocol.set_low_latency(sock)
        camera = FakeCamera() if args.fake else OakCamera(args)
        robot = FakeRobot() if args.fake else RobotIO(args)
        run(args, robot, camera, sock)
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        try:
            if robot is not None:
                robot.close()
        finally:
            try:
                if camera is not None:
                    camera.close()
            finally:
                if sock is not None:
                    sock.close()


if __name__ == "__main__":
    main()
