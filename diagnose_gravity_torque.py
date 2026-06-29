#!/usr/bin/env python3
"""Compare commanded vs motor-reported joint torques during gravity compensation.

Hold the arm at close / far poses and watch J1 (and other joints) in the log.
Ctrl-C sends zero torques and exits.

Example:
  python3 diagnose_gravity_torque.py \\
    --firmware v183 \\
    --software-version S-V1.8-3 \\
    --installation-pos right \\
    --joint-torque-scale 2 1.5 1.5 1 1 1
"""

import argparse
import math
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = THIS_DIR.parent
LOCAL_PYAGXARM = WORKSPACE_DIR / "pyAgxArm"

sys.path.insert(0, str(THIS_DIR))
if LOCAL_PYAGXARM.exists():
    sys.path.insert(0, str(LOCAL_PYAGXARM))

from scripts.piper_api.gravity_compensation import MujocoGravityCompensator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="assets/models/piper_grav_comp_payload_700g.xml",
    )
    parser.add_argument("--firmware", choices=("default", "v183", "v188"), default="v183")
    parser.add_argument("--software-version", default=None)
    parser.add_argument(
        "--installation-pos",
        choices=("horizontal", "left", "right", "unchanged"),
        default="right",
    )
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--torque-scale", type=float, default=1.0)
    parser.add_argument(
        "--joint-torque-scale",
        type=float,
        nargs=6,
        default=[1.0] * 6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    )
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.5,
        help="Seconds between log lines. Default: 0.5.",
    )
    parser.add_argument(
        "--joint",
        type=int,
        default=1,
        choices=range(1, 7),
        help="Joint index to highlight in the summary line. Default: 1.",
    )
    return parser.parse_args()


def fmt(values):
    return " ".join(f"{v:+7.3f}" for v in values)


def main():
    args = parse_args()
    from scripts.piper_api.piper_api import PiperAPI

    compensator = MujocoGravityCompensator(
        args.model_path,
        driver=args.firmware,
        installation_pos=args.installation_pos,
    )
    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
        installation_pos=args.installation_pos,
    )

    sent = False
    ji = args.joint - 1
    try:
        print("Connecting...")
        api.connect()
        version = args.software_version or api.get_software_version()
        compensator.set_firmware_version(version, driver=args.firmware)
        print(f"Software version: {version or 'unknown'}")
        print(f"Firmware driver: {args.firmware}")
        print(f"Joint torque scales: {fmt(args.joint_torque_scale)}")
        print(
            "\nHold the arm still at CLOSE, then FAR pose.\n"
            "Compare cmd vs fb for the highlighted joint.\n"
            "Press Enter to enable MIT torque output (Ctrl-C to abort)..."
        )
        input()

        if not api.enable():
            raise RuntimeError("Timed out enabling Piper")
        api.set_mit_mode()
        print("MIT enabled. Logging...\n")

        next_print = time.monotonic()
        while True:
            joint_state = api.get_joint_state()
            motor_state = api.get_motor_state()
            if joint_state is None:
                raise RuntimeError("Joint feedback unavailable")

            joints = joint_state.joints
            velocities = (
                motor_state.velocities if motor_state is not None else [0.0] * 6
            )
            result = compensator.compute(
                joints,
                joint_velocities=velocities,
                damping=args.damping,
                torque_scale=args.torque_scale,
                joint_torque_scales=args.joint_torque_scale,
            )
            api.move_mit_torques(result.command_torques, p_des=joints, kp=0.0, kd=0.0)
            sent = True

            now = time.monotonic()
            if now >= next_print:
                fb = (
                    motor_state.torques
                    if motor_state is not None
                    else [float("nan")] * 6
                )
                print(f"pose J2={joints[1]:+.3f} J3={joints[2]:+.3f} rad")
                print(f"  gravity: [{fmt(result.gravity_torques)}] N*m")
                print(f"  cmd    : [{fmt(result.command_torques)}] N*m")
                print(f"  fb     : [{fmt(fb)}] N*m")
                print(
                    f"  J{args.joint}  gravity={result.gravity_torques[ji]:+.3f}  "
                    f"cmd={result.command_torques[ji]:+.3f}  fb={fb[ji]:+.3f}  "
                    f"(cmd-fb={result.command_torques[ji]-fb[ji]:+.3f})"
                )
                same_sign = (
                    math.copysign(1, result.command_torques[ji]) == math.copysign(1, fb[ji])
                    if abs(result.command_torques[ji]) > 0.05 and abs(fb[ji]) > 0.05
                    else None
                )
                if same_sign is True:
                    print("  sign: cmd/fb SAME")
                elif same_sign is False:
                    print("  sign: cmd/fb OPPOSITE  <-- likely MIT sign/scaling issue")
                print()
                next_print = now + args.print_every
            time.sleep(0.005)
    finally:
        if sent:
            api.move_mit_torques([0.0] * 6)
            print("Sent zero MIT torques.")
        api.disconnect()


if __name__ == "__main__":
    main()
