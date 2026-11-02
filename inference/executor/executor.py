"""Execute 19D policy chunks on Trimorph + Piper, using collection coordinates.

The request/reply loop is synchronous: prediction latency remains a pause
between chunks. Timings explicitly expose that cost; this is not an async
controller. No startup home motion or leader-to-follower retargeting is used.
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
            self.piper.set_speed_percent(args.piper_speed)
        except BaseException:
            self.close()
            raise

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
        dai = import_depthai()
        require_depthai_device(dai)
        self.pipeline = dai.Pipeline()
        try:
            cam = self.pipeline.create(dai.node.Camera).build()
            apply_camera_control(dai, cam, brightness=args.cam_brightness,
                                 anti_banding_hz=args.cam_anti_banding)
            output = cam.requestOutput((args.cam_width, args.cam_height), dai.ImgFrame.Type.NV12)
            self.queue = output.createOutputQueue()
            self.pipeline.start()
            # Match collection warmup, bounded by the configured timeout.
            self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            packet = self.queue.tryGet()
            if packet is None:
                time.sleep(0.002)
                continue
            # Prefer the newest available image after executing a whole chunk.
            for _ in range(100):
                newer = self.queue.tryGet()
                if newer is None:
                    break
                packet = newer
            bgr = packet.getCvFrame()
            if bgr is not None:
                bgr = self.flip_frame(self.cv2, bgr, self.flip)
                return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        raise TimeoutError("No OAK camera image within camera timeout")

    def close(self):
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
        rtt_ms = (time.perf_counter() - send_start) * 1000
        execute_start = time.perf_counter()
        overruns = 0
        for index, action in enumerate(chunk):
            robot.apply_action(action)
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
    p.add_argument("--hand-api-repo", type=Path, default=CONTROL_ROOT.parent / "trimorph_hand_api")
    p.add_argument("--hand-config", type=Path, default=CONTROL_ROOT / defaults.DEFAULT_HAND_CONFIG)
    p.add_argument("--hand-port", default=defaults.DEFAULT_HAND_PORT)
    p.add_argument("--hand-velocity", type=float, default=defaults.FOLLOWER_PROFILE_VELOCITY_RAD_S)
    p.add_argument("--hand-acceleration", type=float, default=defaults.FOLLOWER_PROFILE_ACCELERATION_RAD_S2)
    for name, default in (
        ("interface", defaults.DEFAULT_PIPER_INTERFACE), ("channel", defaults.DEFAULT_PIPER_CHANNEL),
        ("firmware", defaults.DEFAULT_PIPER_FIRMWARE), ("installation-pos", defaults.DEFAULT_PIPER_INSTALLATION_POS),
        ("payload", defaults.DEFAULT_PIPER_PAYLOAD),
    ):
        p.add_argument(f"--piper-{name}", default=default)
    p.add_argument("--piper-speed", type=int, default=100)
    p.add_argument("--cam-width", type=int, default=defaults.TASK_CAMERA_WIDTH)
    p.add_argument("--cam-height", type=int, default=defaults.TASK_CAMERA_HEIGHT)
    p.add_argument("--cam-flip", default=defaults.TASK_CAMERA_FLIP,
                   choices=["none", "horizontal", "vertical", "both", "180"])
    p.add_argument("--cam-brightness", type=int, default=defaults.TASK_CAMERA_BRIGHTNESS)
    p.add_argument("--cam-anti-banding", default=defaults.TASK_CAMERA_ANTI_BANDING_HZ,
                   choices=["off", "50", "60", "auto"])
    p.add_argument("--camera-timeout", type=float, default=5.0)
    args = p.parse_args(argv)
    for name in ("hz", "timeout", "camera_timeout", "hand_velocity", "hand_acceleration"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.max_cycles < 0 or not 1 <= args.jpeg_quality <= 100 or not 1 <= args.piper_speed <= 100:
        p.error("Require max-cycles >= 0, jpeg-quality 1..100, piper-speed 1..100")
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
