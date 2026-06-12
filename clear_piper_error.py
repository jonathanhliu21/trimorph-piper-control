#!/usr/bin/env python3
"""Clear Piper joint error and re-enable the arm."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.piper_api import PiperAPI

api = PiperAPI()
api.connect()
time.sleep(0.3)

state = api.get_state()
print(f"Before: ctrl_mode={state.get('ctrl_mode')}  arm_status={state.get('arm_status')}  err_code={state.get('err_code')}")

print("Resetting motion controller...")
api.robot.reset()
time.sleep(0.5)

print("Clearing all joint errors...")
api.robot.clear_joint_error(joint_index=255)
time.sleep(1.0)

print("Re-enabling...")
ok = api.enable(timeout_s=5.0)
time.sleep(0.5)

state = api.get_state()
print(f"After:  ctrl_mode={state.get('ctrl_mode')}  arm_status={state.get('arm_status')}  err_code={state.get('err_code')}")
print(f"Enable: {ok}")

joints = api.get_joint_angles()
eef = api.get_eef_pose()
if joints:
    print(f"\nJoints (rad): {[round(j, 4) for j in joints]}")
if eef:
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    print("EEF pose:")
    for label, v in zip(labels, eef):
        print(f"  {label:5s}: {v:.6f}")

# If status is still bad, try a joint-space nudge to j6 to shake it
if "TARGET_POS_EXCEEDS_LIMIT" in str(state.get("arm_status", "")):
    print("\nStatus still fault — trying joint-space move to shake controller...")
    if joints:
        safe = list(joints)
        safe[5] = 0.0  # move j6 to 0 rad
        print(f"Commanding joints: {[round(j, 4) for j in safe]}")
        api.move_joint(safe, mode="smooth")
        time.sleep(3.0)
    state = api.get_state()
    print(f"Final: ctrl_mode={state.get('ctrl_mode')}  arm_status={state.get('arm_status')}  err_code={state.get('err_code')}")

api.disconnect()
