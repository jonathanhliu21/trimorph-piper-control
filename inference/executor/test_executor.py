"""No-hardware executor for testing the predictor link end to end.

Exercises the exact same control loop as the real executor (executor.run), but
substitutes a fake robot (zero state, no-op actuation) and a synthetic camera
that compresses like a real frame (~tens of KB, not JPEG-worst-case noise).

Use this to measure real predictor latency and WiFi round-trip without the
robot or RealSense connected:

    python -u fake_executor.py --predictor 192.168.0.206 --port 5555 --max-cycles 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

EXECUTOR_DIR = Path(__file__).resolve().parent
if str(EXECUTOR_DIR) not in sys.path:
    sys.path.insert(0, str(EXECUTOR_DIR))

from executor import EEF_DIM, HAND_DIM, connect_predictor, run  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="MIDAS executor link test (no hardware).")
    p.add_argument("--predictor", default="192.168.0.206")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--hz", type=float, default=15.0,
                   help="Action application rate (matches the real executor).")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--max-cycles", type=int, default=20)
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    return p.parse_args()


class FakeRobot:
    def __init__(self):
        self._state = np.zeros(EEF_DIM + HAND_DIM, dtype=np.float32)

    def read_state(self) -> np.ndarray:
        return self._state.copy()

    def apply_action(self, action: np.ndarray) -> None:
        self._state = action.astype(np.float32)

    def close(self):
        pass


class FakeCamera:
    """Smooth synthetic frame that compresses like a real photo (~tens of KB),
    so network timings reflect a realistic payload, not JPEG-worst-case noise."""

    def __init__(self, width, height):
        yy, xx = np.meshgrid(
            np.linspace(0, 255, height), np.linspace(0, 255, width), indexing="ij"
        )
        self._base = np.stack([xx, yy, (xx + yy) / 2], axis=-1).astype(np.float32)

    def read(self) -> np.ndarray:
        frame = self._base + np.random.uniform(-4, 4, self._base.shape)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def close(self):
        pass


def main():
    args = parse_args()
    print("[fake] no hardware; testing the predictor link only.")
    robot = FakeRobot()
    camera = FakeCamera(args.cam_width, args.cam_height)
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
