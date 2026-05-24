"""MIDAS action executor (real hardware).

Runs on the robot box. Each cycle (synchronous "stall" design):

    1. read robot state  -> [eef_6, hand_13]  (19,)
    2. capture camera frame (RGB, flipped to match training)
    3. send (image, state) to the predictor and block for the reply
    4. execute the returned absolute action chunk at a fixed rate (--hz)

Observation is captured at the *top* of the loop, i.e. right after the previous
chunk finished executing, so the lag fed to the policy is only the network +
inference time (not a whole chunk).

The action application rate (--hz) must match the rate the policy was trained
at (15Hz by default). For a no-hardware link/latency test, use fake_executor.py.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(INFERENCE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import protocol  # noqa: E402

EEF_DIM = 6
HAND_DIM = 13


def parse_args():
    p = argparse.ArgumentParser(description="MIDAS action executor (real hardware).")
    p.add_argument("--predictor", default="192.168.0.206",
                   help="Predictor (GPU box) LAN IP.")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--hz", type=float, default=15.0,
                   help="Action application rate. Must match the policy's training "
                        "rate (default 15).")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--max-cycles", type=int, default=0,
                   help="Stop after N predict cycles (0 = run until Ctrl-C).")
    # Camera (must match data collection).
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--cam-serial", default=None)
    p.add_argument("--cam-flip", default="both",
                   choices=["none", "horizontal", "vertical", "both"])
    # Hand.
    p.add_argument("--hand-port", default=None)
    p.add_argument("--hand-baudrate", type=int, default=4_000_000)
    p.add_argument("--hand-config", default=None)
    # Piper.
    p.add_argument("--piper-channel", default="can0")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
class RealSenseCamera:
    """RGB8 stream flipped to match the data-collection pipeline."""

    def __init__(self, width, height, fps, serial, flip):
        import cv2
        import pyrealsense2 as rs

        self._cv2 = cv2
        self._flip = flip
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self._pipeline.start(cfg)
        for _ in range(30):  # warmup
            self._pipeline.wait_for_frames(3000)
        print(f"RealSense {width}x{height} rgb8 @{fps} flip={flip}")

    def read(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames(500)
        rgb = np.asanyarray(frames.get_color_frame().get_data())
        flip = {"horizontal": 1, "vertical": 0, "both": -1}.get(self._flip)
        if flip is not None:
            rgb = self._cv2.flip(rgb, flip)
        return np.ascontiguousarray(rgb)

    def close(self):
        self._pipeline.stop()


# --------------------------------------------------------------------------- #
# Robot I/O
# --------------------------------------------------------------------------- #
class RobotIO:
    """Piper EEF + Midas hand. State/action layout: [eef_6, hand_13]."""

    def __init__(self, args):
        from scripts.midas_api import DEFAULT_CONFIG_PATH, connect_hand
        from scripts.piper_api import PiperAPI

        self._piper = PiperAPI()
        self._piper.connect()
        self._piper.enable()

        hand_args = types.SimpleNamespace(
            port=args.hand_port,
            baudrate=args.hand_baudrate,
            config=args.hand_config or str(DEFAULT_CONFIG_PATH),
        )
        self._hand = connect_hand(hand_args, enable_torque=True)
        print(f"Piper on {args.piper_channel}; hand on {self._hand.port} "
              f"({len(self._hand.motor_ids)} motors)")

    def read_state(self) -> np.ndarray:
        eef = self._piper.get_eef_pose()
        hand = self._hand.read_pos()
        if eef is None:
            raise RuntimeError("Piper EEF pose unavailable.")
        state = np.concatenate([
            np.asarray(eef, dtype=np.float32),
            np.asarray(hand, dtype=np.float32),
        ])
        if state.shape[0] != EEF_DIM + HAND_DIM:
            raise RuntimeError(
                f"State dim {state.shape[0]} != {EEF_DIM + HAND_DIM} "
                f"(eef={len(eef)}, hand={len(hand)})."
            )
        return state

    def apply_action(self, action: np.ndarray) -> None:
        self._piper.move_eef(action[:EEF_DIM].tolist())
        self._hand.set_positions(action[EEF_DIM:], clip=True)

    def close(self):
        try:
            self._hand.shutdown()
        finally:
            self._piper.disconnect()


# --------------------------------------------------------------------------- #
# Shared control loop (also used by fake_executor.py)
# --------------------------------------------------------------------------- #
def connect_predictor(host, port) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    protocol.set_low_latency(sock)
    print(f"Connecting to predictor {host}:{port} ...")
    sock.connect((host, port))
    print("Connected.")
    return sock


def run(args, robot, camera, sock):
    dt = 1.0 / args.hz
    seq = 0
    while args.max_cycles == 0 or seq < args.max_cycles:
        state = robot.read_state()
        image_rgb = camera.read()

        t_send = time.perf_counter()
        protocol.send_obs(sock, seq, image_rgb, state,
                          t_capture=time.time(), jpeg_quality=args.jpeg_quality)
        reply = protocol.recv_action(sock)
        if reply is None:
            print("Predictor closed the connection.")
            return
        _rseq, chunk, header = reply
        rtt_ms = (time.perf_counter() - t_send) * 1e3

        # Execute the chunk at a fixed rate using an absolute schedule so timing
        # does not drift across the chunk.
        next_t = time.perf_counter()
        for a in chunk:
            robot.apply_action(a)
            next_t += dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

        print(f"cycle={seq} chunk={chunk.shape[0]} @ {args.hz:g}Hz "
              f"infer={header.get('infer_ms', 0):.1f}ms rtt={rtt_ms:.1f}ms")
        seq += 1


def main():
    args = parse_args()
    robot = RobotIO(args)
    camera = RealSenseCamera(args.cam_width, args.cam_height, args.cam_fps,
                             args.cam_serial, args.cam_flip)
    sock = connect_predictor(args.predictor, args.port)
    try:
        run(args, robot, camera, sock)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sock.close()
        camera.close()
        robot.close()


if __name__ == "__main__":
    main()
