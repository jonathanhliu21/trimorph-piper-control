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
HAND_CURRENT_LIMIT_MA = 500
HAND_PROFILE_VELOCITY_RAD_S: float | None = 3.0
HAND_PROFILE_ACCELERATION_RAD_S2: float | None = 50.0

# Set to None to skip moving the arm at startup (stays wherever it currently is).
EEF_HOME: np.ndarray | None = None

HAND_HOME = np.array([
    -0.062922,  # pos_0
     0.003141,  # pos_1
     0.322954,  # pos_2
     0.188665,  # pos_3
    -0.032484,  # pos_4
    -0.043237,  # pos_5
    -0.046735,  # pos_6
    -0.105224,  # pos_7
    -0.122916,  # pos_8
     0.005457,  # pos_9
    -0.043675,  # pos_10
    -0.042769,  # pos_11
     0.115910,  # pos_12
], dtype=np.float32)


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
        state = self._piper.get_state()
        print(f"Piper state at connect: ctrl_mode={state.get('ctrl_mode')} "
              f"arm_status={state.get('arm_status')} err_code={state.get('err_code')}")
        if not self._piper.enable():
            print("Warning: Piper enable timed out — arm may not respond to commands.")
        time.sleep(0.5)  # let the controller settle into autonomous mode
        state = self._piper.get_state()
        print(f"Piper state after enable: ctrl_mode={state.get('ctrl_mode')} "
              f"arm_status={state.get('arm_status')} err_code={state.get('err_code')}")

        hand_args = types.SimpleNamespace(
            port=args.hand_port,
            baudrate=args.hand_baudrate,
            config=args.hand_config or str(DEFAULT_CONFIG_PATH),
        )
        self._hand = connect_hand(hand_args, enable_torque=True)
        self._hand.set_current_limit(HAND_CURRENT_LIMIT_MA)
        self._hand.set_motion_profile(
            profile_velocity_rad_s=HAND_PROFILE_VELOCITY_RAD_S,
            profile_acceleration_rad_s2=HAND_PROFILE_ACCELERATION_RAD_S2,
        )
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

    def apply_action(self, action: np.ndarray, *, _debug_once: list = [False]) -> None:
        eef = action[:EEF_DIM].copy()
        eef[4] = np.clip(eef[4], -np.pi / 2, np.pi / 2)  # pitch limit
        if not _debug_once[0]:
            current = self._piper.get_eef_pose() or []
            print(f"First EEF command: {[round(v,4) for v in eef.tolist()]}")
            print(f"Current EEF state: {[round(v,4) for v in current]}")
            _debug_once[0] = True
        self._piper.move_eef(eef.tolist())
        self._hand.set_positions(action[EEF_DIM:], clip=True)

    def close(self):
        try:
            self._hand.shutdown()
        finally:
            self._piper.disconnect()


# --------------------------------------------------------------------------- #
# Shared control loop (also used by fake_executor.py)
# --------------------------------------------------------------------------- #
def move_to_home(robot: RobotIO, duration_s: float = 3.0, hz: float = 50.0,
                 arm_settle_s: float = 5.0, arm_speed_percent: int = 50) -> None:
    """Move arm and hand to their training-average home positions.

    The Piper command is sent immediately (it has its own motion controller);
    the hand is interpolated over duration_s. Afterward we wait up to
    arm_settle_s for the arm to converge within 1 cm / 0.05 rad.
    """
    print(f"Moving to home position ({'hand only' if EEF_HOME is None else 'arm + hand'}) ...")

    robot._piper.set_speed_percent(arm_speed_percent)
    hand_start = np.asarray(robot._hand.read_pos(), dtype=np.float32)
    n_steps = max(1, int(duration_s * hz))
    dt = duration_s / n_steps
    for i in range(1, n_steps + 1):
        if EEF_HOME is not None:
            robot._piper.move_eef(EEF_HOME.tolist())
        alpha = i / n_steps
        robot._hand.set_positions(hand_start + alpha * (HAND_HOME - hand_start), clip=True)
        time.sleep(dt)

    if EEF_HOME is not None:
        deadline = time.perf_counter() + arm_settle_s
        while time.perf_counter() < deadline:
            robot._piper.move_eef(EEF_HOME.tolist())
            eef = robot._piper.get_eef_pose()
            if eef is not None:
                err = np.asarray(eef, dtype=np.float32) - EEF_HOME
                if np.linalg.norm(err[:3]) < 0.01 and np.linalg.norm(err[3:]) < 0.05:
                    break
            time.sleep(0.05)
    print("At home position.")


def connect_predictor(host, port) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    protocol.set_low_latency(sock)
    print(f"Connecting to predictor {host}:{port} ...")
    sock.connect((host, port))
    print("Connected.")
    return sock


def run(args, robot, camera, sock, chunks: list[np.ndarray]) -> None:
    import cv2
    dt = 1.0 / args.hz
    seq = 0
    while args.max_cycles == 0 or seq < args.max_cycles:
        state = robot.read_state()
        image_rgb = camera.read()

        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("wrist camera", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit key pressed.")
            break

        t_send = time.perf_counter()
        protocol.send_obs(sock, seq, image_rgb, state,
                          t_capture=time.time(), jpeg_quality=args.jpeg_quality)
        reply = protocol.recv_action(sock)
        if reply is None:
            print("Predictor closed the connection.")
            return
        _rseq, chunk, header = reply
        rtt_ms = (time.perf_counter() - t_send) * 1e3
        chunks.append(chunk.copy())

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


EEF_LABELS = ["x", "y", "z", "roll", "pitch", "yaw"]
HAND_LABELS = [f"pos_{i}" for i in range(HAND_DIM)]


def plot_actions(chunks: list[np.ndarray]) -> None:
    if not chunks:
        print("No actions recorded — nothing to plot.")
        return
    import matplotlib.pyplot as plt

    actions = np.concatenate(chunks, axis=0)  # (T, 19)
    eef = actions[:, :EEF_DIM]
    hand = actions[:, EEF_DIM:]
    t = np.arange(len(actions))

    fig, (ax_eef, ax_hand) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i, label in enumerate(EEF_LABELS):
        ax_eef.plot(t, eef[:, i], label=label)
    ax_eef.set_ylabel("EEF action")
    ax_eef.legend(ncol=6, fontsize=8)
    ax_eef.grid(True)

    for i, label in enumerate(HAND_LABELS):
        ax_hand.plot(t, hand[:, i], label=label)
    ax_hand.set_ylabel("Hand action (rad)")
    ax_hand.set_xlabel("Action step")
    ax_hand.legend(ncol=7, fontsize=8)
    ax_hand.grid(True)

    fig.suptitle(f"Predictor actions  ({len(chunks)} chunks, {len(actions)} steps)")
    plt.tight_layout()

    save_path = Path(__file__).parent / "action_log.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved action plot to {save_path}")


def main():
    args = parse_args()
    robot = RobotIO(args)
    move_to_home(robot)
    camera = RealSenseCamera(args.cam_width, args.cam_height, args.cam_fps,
                             args.cam_serial, args.cam_flip)
    sock = connect_predictor(args.predictor, args.port)
    chunks: list[np.ndarray] = []
    try:
        run(args, robot, camera, sock, chunks)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sock.close()
        camera.close()
        robot.close()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
    plot_actions(chunks)


if __name__ == "__main__":
    main()
