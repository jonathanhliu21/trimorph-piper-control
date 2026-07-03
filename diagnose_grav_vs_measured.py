#!/usr/bin/env python3
"""Compare the MuJoCo model's gravity torque g(q) against the REAL holding
torque the firmware reports, joint-by-joint, at several static poses.

Why: in pure gravity-comp teach mode (kp=0, no friction comp) the arm receives
exactly the model's g(q). If the model matches reality the arm holds still;
drift/droop means model g(q) != real g(q). This tool measures the gap per joint
so we stop guessing which joint (and how much) is wrong.

How it works: the arm is slewed via firmware POSITION mode to each pose and held.
While holding, the firmware-reported motor torque (`motor_state.torques`) balances
gravity + static friction, so it is the physical joint torque in N*m -- directly
comparable to the model's raw g(q) (same sign convention that calibrate_gravity.py
already relies on). The difference (real - model) is the model error; static
friction (~0.1-0.2 N*m) sets the noise floor, so only larger, systematic
residuals indicate a genuine model/CoM mismatch.

This does NOT command any torque -- it only reads feedback while the firmware
holds each pose, so it is as safe as calibrate_gravity.py's capture.

Usage:
    python3 diagnose_grav_vs_measured.py --firmware v183 --gravity 0.240 -9.800 -0.375
"""

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = THIS_DIR.parent
LOCAL_PYAGXARM = WORKSPACE_DIR / "pyAgxArm"

sys.path.insert(0, str(THIS_DIR))
if LOCAL_PYAGXARM.exists():
    sys.path.insert(0, str(LOCAL_PYAGXARM))

from scripts.piper_api.gravity_compensation import MujocoGravityCompensator

# Reuse the tested hardware-motion + capture code from the calibration tool.
from calibrate_gravity import _capture_samples

DEFAULT_MODEL_PATH = THIS_DIR / "assets" / "models" / "midas_piper_grav_comp.xml"

FRICTION_FLOOR_NM = 0.2  # residuals below this are plausibly just static friction


def _diag_offsets(d):
    """Perturbations exciting BOTH the base (J1-J3) and wrist (J4-J6), so every
    joint's gravity torque is probed, not just one group."""
    return [
        [0, 0, 0, 0, 0, 0],
        [+d, 0, 0, 0, 0, 0],
        [-d, 0, 0, 0, 0, 0],
        [0, +d, -d / 2, 0, 0, 0],
        [0, -d, +d / 2, 0, 0, 0],
        [0, 0, 0, +d, 0, 0],
        [0, 0, 0, -d, 0, 0],
        [0, 0, 0, 0, +d, 0],
        [0, 0, 0, 0, -d, 0],
        [0, 0, 0, 0, 0, +d],
        [0, 0, 0, 0, 0, -d],
        [0, +d / 2, -d / 4, +d, +d, +d],
        [0, -d / 2, +d / 4, -d, -d, -d],
    ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gravity", type=float, nargs=3, required=True,
                   metavar=("GX", "GY", "GZ"),
                   help="Gravity vector fed to the model (use the same value you "
                        "pass to test_mit_controller.py, e.g. 0.240 -9.800 -0.375).")
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--firmware", choices=("default", "v183", "v188"), default="v183")
    p.add_argument("--channel", "--can-port", dest="channel", default="can0")
    p.add_argument("--interface", default="socketcan")
    p.add_argument("--installation-pos",
                   choices=("horizontal", "left", "right", "unchanged"),
                   default="unchanged")
    p.add_argument("--speed-percent", type=int, default=5)
    p.add_argument("--perturb", type=float, default=0.15,
                   help="Pose perturbation amplitude (rad). Default: 0.15.")
    p.add_argument("--step-rad", type=float, default=0.03)
    p.add_argument("--step-dt", type=float, default=0.08)
    p.add_argument("--auto", action="store_true",
                   help="Skip per-pose Enter confirmation (NOT recommended).")
    p.add_argument("--settle-s", type=float, default=1.5)
    p.add_argument("--average-s", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    from scripts.piper_api.piper_api import PiperAPI

    comp = MujocoGravityCompensator(args.model_path, gravity=args.gravity)
    print(f"Model:   {args.model_path}")
    print(f"Gravity: {tuple(round(g, 4) for g in comp.get_gravity())}")

    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
        installation_pos=args.installation_pos,
        payload="unchanged",
    )
    try:
        print(f"Connecting on {args.interface}:{args.channel} ...")
        api.connect()
        api.set_speed_percent(args.speed_percent)
        start = api.get_joint_state()
        if start is None:
            raise RuntimeError("No joint feedback; cannot diagnose")
        q0 = np.array(start.joints, dtype=float)
        print(f"Start pose: {np.round(q0, 3).tolist()}")

        samples = _capture_samples(api, q0, _diag_offsets(args.perturb), args)
    finally:
        api.disconnect()

    if not samples:
        raise RuntimeError("No poses captured")

    # Build per-pose comparison: real (firmware) vs model g(q).
    residuals = []  # (real - model) per joint, stacked over poses
    print("\n" + "=" * 78)
    print("PER-POSE: real (firmware holding torque) vs model g(q)  [N*m]")
    print("=" * 78)
    for k, (q, tau_real) in enumerate(samples):
        g_model = comp.compute(q).gravity_torques
        diff = [r - m for r, m in zip(tau_real, g_model)]
        residuals.append(diff)
        print(f"\npose {k}  q = {[round(v, 3) for v in q]}")
        print("        " + "".join(f"   J{j + 1:<7}" for j in range(6)))
        print("  real  " + "".join(f"{tau_real[j]:+8.3f} " for j in range(6)))
        print("  model " + "".join(f"{g_model[j]:+8.3f} " for j in range(6)))
        print("  diff  " + "".join(f"{diff[j]:+8.3f} " for j in range(6)))

    res = np.array(residuals)  # (n_poses, 6)
    mean_res = res.mean(axis=0)
    rms_res = np.sqrt((res ** 2).mean(axis=0))
    max_abs = np.abs(res).max(axis=0)

    print("\n" + "=" * 78)
    print("SUMMARY: model error per joint  (real - model), over all poses [N*m]")
    print("=" * 78)
    print("        " + "".join(f"   J{j + 1:<7}" for j in range(6)))
    print("  mean  " + "".join(f"{mean_res[j]:+8.3f} " for j in range(6)))
    print("  rms   " + "".join(f"{rms_res[j]:+8.3f} " for j in range(6)))
    print("  max|.|" + "".join(f"{max_abs[j]:+8.3f} " for j in range(6)))

    print(f"\nInterpretation (static-friction floor ~{FRICTION_FLOOR_NM:.2f} N*m):")
    flagged = False
    for j in range(6):
        if rms_res[j] > FRICTION_FLOOR_NM:
            flagged = True
            sign = "under-compensates (arm falls)" if mean_res[j] > 0 else \
                   "over-compensates (arm pushed past hold)"
            print(f"  J{j + 1}: rms {rms_res[j]:.3f} N*m > floor  ->  model {sign} "
                  f"by ~{mean_res[j]:+.3f} N*m on average")
    if not flagged:
        print("  All joints within the friction floor -- the model g(q) matches the "
              "measured holding torque well. Drift is then a delivery issue "
              "(raise --torque-scale), not a model error.")
    else:
        print("\n  A joint flagged above is where the model g(q) is wrong -- that "
              "residual, unopposed at kp=0, is what drives the drift/droop there.")
    print()


if __name__ == "__main__":
    main()
