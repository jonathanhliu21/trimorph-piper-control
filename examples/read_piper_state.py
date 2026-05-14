import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.piper_api import PiperAPI  # noqa: E402


def main():
    api = PiperAPI()
    api.connect()

    try:
        print("Connected OK:", api.is_ok())
        print("Receive FPS:", api.get_fps())

        while True:
            joints = api.get_joint_state()
            flange = api.get_eef_state()

            if joints is not None:
                print("joint angles [rad]:", joints.joints)
                print("joint timestamp:", joints.timestamp, "hz:", joints.hz)

            if flange is not None:
                x, y, z, roll, pitch, yaw = flange.pose
                print("flange pose [m, rad]:", flange.pose)
                print(
                    f"x={x:.4f}, y={y:.4f}, z={z:.4f}, "
                    f"rpy=({roll:.4f}, {pitch:.4f}, {yaw:.4f})"
                )
                print("flange timestamp:", flange.timestamp, "hz:", flange.hz)

            print()
            time.sleep(0.05)

    finally:
        api.disconnect()


if __name__ == "__main__":
    main()
