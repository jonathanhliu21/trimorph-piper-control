#!/usr/bin/env python3
"""Calibrate the MuJoCo gravity vector for a (possibly side/wall-mounted) Piper.

The gravity generalized force is linear in the world gravity vector ``g``::

    tau_grav(q) = A(q) @ g          # A(q) is 6x3, g is (gx, gy, gz)

We sample the arm at several static poses, read the measured joint torques
(N*m, which already balance gravity + friction while holding still), build the
stacked system ``tau_meas = A_stack @ g`` and solve for ``g`` by least squares.

Only the *direction* of ``g`` matters for picking the right installation /
``--gravity`` argument, so the result is renormalised to 9.81 m/s^2. This makes
the estimate robust to any constant scale/units mismatch in the torque feedback.

Run ``--self-test`` to validate the math offline (no hardware needed).
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = THIS_DIR.parent
LOCAL_PYAGXARM = WORKSPACE_DIR / "pyAgxArm"

sys.path.insert(0, str(THIS_DIR))
if LOCAL_PYAGXARM.exists():
    sys.path.insert(0, str(LOCAL_PYAGXARM))

from scripts.piper_api.gravity_compensation import (  # type: ignore[import-not-found]
    DEFAULT_JOINT_NAMES,
    MujocoGravityCompensator,
)

G = 9.81

# NOTE: get_payload_700g_model_path() resolves to the (broken) midas_hand model,
# so point at the standalone payload model that actually compiles.
DEFAULT_MODEL_PATH = (
    THIS_DIR / "assets" / "models" / "piper_grav_comp_payload_700g.xml"
)


class GravityModel:
    """Thin wrapper to compute the 6x3 gravity-jacobian A(q) from a MuJoCo model."""

    def __init__(self, model_path, joint_names):
        # Reuse the compensator only to load/validate the model + joint map.
        self._comp = MujocoGravityCompensator(model_path, joint_names=joint_names)
        self._mj = self._comp.mujoco
        self._model = self._comp.model
        self._data = self._comp.data
        self._qpos_idx = self._comp._qpos_indices
        self._qvel_idx = self._comp._qvel_indices

    def gravity_jacobian(self, q):
        """Return A(q): column j is the joint torque vector for unit gravity e_j."""
        cols = []
        for axis in range(3):
            grav = [0.0, 0.0, 0.0]
            grav[axis] = 1.0
            self._model.opt.gravity[:] = grav
            self._data.qpos[:] = self._model.qpos0
            self._data.qvel[:] = 0.0
            self._data.qacc[:] = 0.0
            for idx, val in zip(self._qpos_idx, q):
                self._data.qpos[idx] = val
            self._mj.mj_forward(self._model, self._data)
            cols.append([float(self._data.qfrc_bias[i]) for i in self._qvel_idx])
        # cols is 3 lists of len-6 -> A is 6x3
        return np.array(cols).T

    # ---- payload (end-effector) CoM calibration helpers ----
    def _bias_at(self, q, gravity):
        self._model.opt.gravity[:] = gravity
        self._data.qpos[:] = self._model.qpos0
        self._data.qvel[:] = 0.0
        self._data.qacc[:] = 0.0
        for idx, val in zip(self._qpos_idx, q):
            self._data.qpos[idx] = val
        self._mj.mj_forward(self._model, self._data)
        return np.array([float(self._data.qfrc_bias[i]) for i in self._qvel_idx])

    def payload_model(self, payload_body):
        bid = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_BODY, payload_body)
        if bid < 0:
            raise ValueError(f"payload body {payload_body!r} not in model")
        return bid, dict(
            mass=float(self._model.body_mass[bid]),
            ipos=self._model.body_ipos[bid].copy(),
            pos=self._model.body_pos[bid].copy(),
        )

    def _bias_with_payload(self, q, gravity, bid, mass, pos):
        """Bias torque with the payload set to (mass, CoM=pos in gripper frame).

        CoM is set via body_pos because body_ipos is not honoured at runtime;
        the payload body's own ipos is zero so CoM == body_pos.
        """
        self._model.body_mass[bid] = mass
        self._model.body_ipos[bid] = (0.0, 0.0, 0.0)
        self._model.body_pos[bid] = pos
        return self._bias_at(q, gravity)

    def com_jacobian(self, q, gravity, payload_body, mass):
        """Return (D, ref) for solving the payload CoM offset r at known ``mass``.

        Gravity torque is affine in r: tau(r) = ref + D @ r, where
        ref = bias(mass, r=0) and D[:,k] = bias(mass, e_k) - ref. The constant
        "mass-at-origin" term is captured by ``ref`` (depends on known mass), so
        it does NOT contaminate the columns (unlike a first-moment-only fit).
        """
        bid, orig = self.payload_model(payload_body)
        try:
            ref = self._bias_with_payload(q, gravity, bid, mass, (0.0, 0.0, 0.0))
            cols = []
            for axis in range(3):
                pos = [0.0, 0.0, 0.0]
                pos[axis] = 1.0
                cols.append(self._bias_with_payload(q, gravity, bid, mass, pos) - ref)
        finally:
            self._model.body_mass[bid] = orig["mass"]
            self._model.body_pos[bid] = orig["pos"]
            self._model.body_ipos[bid] = orig["ipos"]
        return np.array(cols).T, ref


def solve_payload(model, samples, gravity, payload_body, hand_mass):
    """Least-squares solve the payload CoM offset r (gripper frame) at known mass.

    tau_meas(q) = ref(q) + D(q) @ r   with ref = bias(mass, r=0)
    -> solve (tau_meas - ref) = D @ r. first moment L = mass * r.
    """
    gravity = np.asarray(gravity, dtype=float)
    D_rows, rhs_rows = [], []
    for q, tau in samples:
        D, ref = model.com_jacobian(q, gravity, payload_body, hand_mass)
        D_rows.append(D)
        rhs_rows.append(np.asarray(tau, dtype=float) - ref)
    D_stack = np.vstack(D_rows)
    rhs = np.concatenate(rhs_rows)
    r, _res, _rank, sv = np.linalg.lstsq(D_stack, rhs, rcond=None)
    fit = D_stack @ r
    rms = float(np.sqrt(np.mean((fit - rhs) ** 2)))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf
    return {
        "first_moment_kg_m": float(hand_mass) * r,
        "hand_mass_kg": float(hand_mass),
        "com_in_gripper_m": r,
        "rms_residual_nm": rms,
        "condition_number": cond,
    }


def solve_gravity(model: GravityModel, samples):
    """Least-squares solve for g from [(q, tau_meas), ...]. Returns dict report."""
    A_rows = []
    tau_rows = []
    for q, tau in samples:
        A_rows.append(model.gravity_jacobian(q))
        tau_rows.append(np.asarray(tau, dtype=float))
    A_stack = np.vstack(A_rows)          # (6N, 3)
    tau_stack = np.concatenate(tau_rows)  # (6N,)

    g_raw, _residual, _rank, sv = np.linalg.lstsq(A_stack, tau_stack, rcond=None)
    return _build_report(A_stack, tau_stack, g_raw, sv)


def _build_report(A_stack, tau_stack, g_raw, singular_values):
    norm = float(np.linalg.norm(g_raw))
    if norm < 1e-9:
        raise RuntimeError("Degenerate solve: gravity estimate is ~zero")
    direction = g_raw / norm
    g_dir = direction * G                       # renormalised to 9.81
    fit = A_stack @ g_raw
    rms = float(np.sqrt(np.mean((fit - tau_stack) ** 2)))
    cond = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else math.inf
    return {
        "g_raw": g_raw,
        "g_dir": g_dir,
        "raw_magnitude": norm,
        "rms_residual_nm": rms,
        "condition_number": cond,
    }


def nearest_installation(g_dir):
    """Describe g_dir relative to base axes and the in-plane angle theta (X-Y)."""
    gx, gy, gz = g_dir
    theta_deg = math.degrees(math.atan2(gy, gx))
    z_frac = abs(gz) / G
    return theta_deg, z_frac


def print_report(report):
    g = report["g_dir"]
    print("\n========== GRAVITY CALIBRATION RESULT ==========")
    print(
        f"Estimated gravity direction (m/s^2): "
        f"{g[0]:+.3f} {g[1]:+.3f} {g[2]:+.3f}"
    )
    print(f"Raw fitted |g| (sanity, ~9.81 if torque units N*m): "
          f"{report['raw_magnitude']:.3f}")
    print(f"Fit RMS residual: {report['rms_residual_nm']:.4f} N*m "
          f"(lower = better; friction sets the floor)")
    print(f"Condition number: {report['condition_number']:.1f} "
          f"(lower = better-conditioned poses)")
    theta_deg, z_frac = nearest_installation(g)
    print(f"In base X-Y plane: angle theta = {theta_deg:+.1f} deg from +X axis")
    print(f"Out-of-plane (|gz|/g): {z_frac:.3f} "
          f"({'≈in-plane (wall mount)' if z_frac < 0.15 else 'has Z tilt'})")
    print("\nUse this directly:")
    print(f"  --gravity {g[0]:.3f} {g[1]:.3f} {g[2]:.3f}")
    print("===============================================\n")


# --------------------------------------------------------------------------- #
# Self-test: validate the solver with synthetic data (no hardware).
# --------------------------------------------------------------------------- #
def run_self_test(args):
    model = GravityModel(args.model_path, args.joint_names)
    rng = np.random.default_rng(0)
    true_g = np.array(args.self_test_gravity, dtype=float)
    print(f"Self-test: ground-truth gravity = {true_g}")

    # Diverse poses exciting J1/J2/J3.
    base = np.array([0.0, 0.4, -0.4, 0.0, 0.3, 0.0])
    poses = []
    for dj1 in (-0.6, 0.0, 0.6):
        for dj2 in (-0.4, 0.0, 0.4):
            q = base.copy()
            q[0] += dj1
            q[1] += dj2
            q[2] -= dj2 * 0.5
            poses.append(q.tolist())

    samples = []
    for q in poses:
        A = model.gravity_jacobian(q)
        tau = A @ true_g
        tau = tau + rng.normal(0.0, args.self_test_friction, size=6)  # friction noise
        samples.append((q, tau.tolist()))

    report = solve_gravity(model, samples)
    print_report(report)
    err = np.degrees(
        math.acos(
            np.clip(
                np.dot(report["g_dir"], true_g)
                / (np.linalg.norm(report["g_dir"]) * np.linalg.norm(true_g)),
                -1.0,
                1.0,
            )
        )
    )
    print(f"Angular error vs ground truth: {err:.2f} deg")
    if err < 5.0:
        print("SELF-TEST PASS (direction recovered within 5 deg).")
    else:
        print("SELF-TEST WARN (direction error > 5 deg; check conditioning/noise).")


# --------------------------------------------------------------------------- #
# Hardware capture: firmware position-hold at perturbed poses, read torques.
# --------------------------------------------------------------------------- #
def _slew_to(api, target, *, step_rad, step_dt):
    """Move to target gradually in small joint steps so motion stays slow.

    Each commanded waypoint differs from the previous by at most ``step_rad`` on
    the largest joint, so the arm creeps to the goal regardless of the firmware
    speed setting. Ctrl-C during a slew leaves the arm where it is.
    """
    js = api.get_joint_state()
    if js is None:
        raise RuntimeError("No joint feedback during slew")
    cur = np.array(js.joints, dtype=float)
    tgt = np.array(target, dtype=float)
    max_delta = float(np.max(np.abs(tgt - cur)))
    n = max(1, int(math.ceil(max_delta / max(step_rad, 1e-6))))
    for i in range(1, n + 1):
        waypoint = cur + (tgt - cur) * (i / n)
        api.move_joint(waypoint.tolist(), mode="exact")
        time.sleep(step_dt)


def _gravity_offsets(d):
    """Perturbations exciting gravity along all base axes (vary J1-J3)."""
    return [
        [0, 0, 0, 0, 0, 0],
        [+d, 0, 0, 0, 0, 0],
        [-d, 0, 0, 0, 0, 0],
        [0, +d, -d / 2, 0, 0, 0],
        [0, -d, +d / 2, 0, 0, 0],
        [+d, +d, -d / 2, 0, 0, 0],
        [-d, -d, +d / 2, 0, 0, 0],
    ]


def _payload_offsets(d):
    """Perturbations exciting the wrist CoM (vary J4-J6, plus J2 for J2/J3 load)."""
    return [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, +d, 0, 0],
        [0, 0, 0, -d, 0, 0],
        [0, 0, 0, 0, +d, 0],
        [0, 0, 0, 0, -d, 0],
        [0, 0, 0, 0, 0, +d],
        [0, 0, 0, 0, 0, -d],
        [0, +d / 2, 0, +d, +d, +d],
        [0, +d / 2, 0, -d, -d, -d],
    ]


def _capture_samples(api, q0, offsets, args):
    """Slew to each perturbed pose, hold, average measured (q, torque)."""
    samples = []
    print(
        f"\nSAFETY: {len(offsets)} poses. Each move is slewed in "
        f"{args.step_rad:.3f} rad steps (max ±{args.perturb:.2f} rad), slowly.\n"
        f"You confirm BEFORE every move; Ctrl-C aborts at any time."
    )
    input("Press Enter to enable and begin calibration... ")
    if not api.enable():
        raise RuntimeError("Timed out enabling Piper")

    for k, off in enumerate(offsets):
        target = (q0 + np.array(off, dtype=float)).tolist()
        if not args.auto:
            ans = input(
                f"\nPose {k}/{len(offsets) - 1}: move to "
                f"{np.round(target, 3).tolist()} ? [Enter=go, s=skip, q=quit] "
            ).strip().lower()
            if ans == "q":
                print("Aborting on user request.")
                break
            if ans == "s":
                print("  skipped")
                continue
        _slew_to(api, target, step_rad=args.step_rad, step_dt=args.step_dt)
        time.sleep(args.settle_s)

        taus, qs = [], []
        t_end = time.monotonic() + args.average_s
        while time.monotonic() < t_end:
            ms = api.get_motor_state()
            js = api.get_joint_state()
            if ms is not None and js is not None:
                taus.append(ms.torques)
                qs.append(js.joints)
            time.sleep(0.01)
        if not taus:
            print(f"  pose {k}: no torque feedback, skipped")
            continue
        tau_avg = np.mean(np.array(taus), axis=0).tolist()
        q_avg = np.mean(np.array(qs), axis=0).tolist()
        samples.append((q_avg, tau_avg))
        print(f"  pose {k}: tau = " + " ".join(f"{t:+.3f}" for t in tau_avg))

    print("\nReturning to start pose...")
    _slew_to(api, q0.tolist(), step_rad=args.step_rad, step_dt=args.step_dt)
    time.sleep(args.settle_s)
    return samples


def run_hardware(args):
    from scripts.piper_api.piper_api import PiperAPI

    if args.calibrate == "payload":
        if args.gravity is None:
            raise SystemExit(
                "--calibrate payload requires --gravity GX GY GZ "
                "(use the value from the gravity calibration)."
            )
        if args.hand_mass is None:
            raise SystemExit("--calibrate payload requires --hand-mass KG")

    model = GravityModel(args.model_path, args.joint_names)
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
            raise RuntimeError("No joint feedback; cannot calibrate")
        q0 = np.array(start.joints, dtype=float)
        print(f"Start pose: {np.round(q0, 3).tolist()}")

        offsets = (
            _payload_offsets(args.perturb)
            if args.calibrate == "payload"
            else _gravity_offsets(args.perturb)
        )
        samples = _capture_samples(api, q0, offsets, args)
    finally:
        api.disconnect()

    if len(samples) < 4:
        raise RuntimeError(f"Need >=4 good poses, got {len(samples)}")

    if args.calibrate == "payload":
        report = solve_payload(
            model, samples, args.gravity, args.payload_body, args.hand_mass
        )
        print_payload_report(report, args.payload_body)
    else:
        print_report(solve_gravity(model, samples))


def print_payload_report(report, payload_body):
    com = report["com_in_gripper_m"]
    L = report["first_moment_kg_m"]
    print("\n========== PAYLOAD CoM CALIBRATION RESULT ==========")
    print(f"Hand mass (given): {report['hand_mass_kg']:.4f} kg")
    print(
        f"CoM offset in gripper frame (m): "
        f"{com[0]:+.4f} {com[1]:+.4f} {com[2]:+.4f}"
    )
    print(
        f"First moment m*r (kg*m): {L[0]:+.5f} {L[1]:+.5f} {L[2]:+.5f}"
    )
    print(f"Fit RMS residual: {report['rms_residual_nm']:.4f} N*m")
    print(f"Condition number: {report['condition_number']:.1f}")
    print(f"\nUpdate the model: set the {payload_body!r} body to")
    print(f'  pos="{com[0]:.4f} {com[1]:.4f} {com[2]:.4f}" '
          f'mass="{report["hand_mass_kg"]:.4f}"')
    print("===================================================\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--calibrate", choices=("gravity", "payload"), default="gravity",
        help=(
            "What to estimate. 'gravity': mount gravity direction (vary J1-J3). "
            "'payload': end-effector CoM offset at known mass (vary J4-J6); "
            "requires --gravity and --hand-mass."
        ),
    )
    p.add_argument(
        "--gravity", type=float, nargs=3, default=None, metavar=("GX", "GY", "GZ"),
        help="Calibrated gravity vector, required for --calibrate payload.",
    )
    p.add_argument(
        "--hand-mass", type=float, default=None, metavar="KG",
        help="Known end-effector mass (kg), required for --calibrate payload.",
    )
    p.add_argument(
        "--payload-body", default="payload_tool",
        help="MuJoCo body name of the end-effector payload. Default: payload_tool.",
    )
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument(
        "--joint-names", nargs=6, default=list(DEFAULT_JOINT_NAMES),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    )
    p.add_argument("--firmware", choices=("default", "v183", "v188"), default="v183")
    p.add_argument("--channel", "--can-port", dest="channel", default="can0")
    p.add_argument("--interface", default="socketcan")
    p.add_argument(
        "--installation-pos",
        choices=("horizontal", "left", "right", "unchanged"),
        default="unchanged",
        help="Firmware install profile sent on connect (does NOT affect the solve).",
    )
    p.add_argument("--speed-percent", type=int, default=5,
                   help="Firmware joint speed cap. Default: 5 (very slow).")
    p.add_argument("--perturb", type=float, default=0.15,
                   help="Pose perturbation amplitude in rad on J1-J3. Default: 0.15.")
    p.add_argument("--step-rad", type=float, default=0.03,
                   help="Max joint step per slew waypoint (rad). Default: 0.03.")
    p.add_argument("--step-dt", type=float, default=0.08,
                   help="Delay between slew waypoints (s). Default: 0.08.")
    p.add_argument("--auto", action="store_true",
                   help="Skip the per-pose Enter confirmation (NOT recommended).")
    p.add_argument("--settle-s", type=float, default=1.5,
                   help="Settle time after each move before reading. Default: 1.5.")
    p.add_argument("--average-s", type=float, default=0.5,
                   help="Torque averaging window per pose. Default: 0.5.")
    p.add_argument("--self-test", action="store_true",
                   help="Validate the solver offline with synthetic data.")
    p.add_argument("--self-test-gravity", type=float, nargs=3,
                   default=[9.81, 0.0, 0.0],
                   help="Ground-truth gravity for --self-test. Default: 9.81 0 0.")
    p.add_argument("--self-test-friction", type=float, default=0.15,
                   help="Synthetic friction noise std (N*m) for --self-test.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_self_test(args)
    else:
        run_hardware(args)


if __name__ == "__main__":
    main()
