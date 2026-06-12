#!/usr/bin/env python3
"""Continuously print Piper joint angles and EEF pose. Ctrl-C to exit."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.piper_api import PiperAPI

api = PiperAPI()
api.connect()
time.sleep(0.3)

try:
    while True:
        joints = api.get_joint_angles()
        eef = api.get_eef_pose()

        lines = []
        if joints:
            for i, v in enumerate(joints):
                lines.append(f"  j{i+1}: {v:+.4f} rad  ({v*180/3.14159:+7.2f}°)")
        else:
            lines.append("  joints unavailable")

        lines.append("")
        if eef:
            for label, v in zip(["x", "y", "z", "roll", "pitch", "yaw"], eef):
                lines.append(f"  {label:5s}: {v:+.4f}")
        else:
            lines.append("  eef unavailable")

        # Clear screen and reprint
        print("\033[2J\033[H", end="")
        print("Piper live readings  (Ctrl-C to quit)\n")
        print("\n".join(lines))
        time.sleep(0.1)

except KeyboardInterrupt:
    pass
finally:
    api.disconnect()
    print("\nDone.")
