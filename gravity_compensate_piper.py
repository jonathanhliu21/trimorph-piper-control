#!/usr/bin/env python3
"""Run MuJoCo-based gravity compensation for an AgileX Piper arm."""

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

from scripts.piper_api.gravity_compensation import (  # type: ignore[import-not-found]
    DEFAULT_JOINT_NAMES,
    MujocoGravityCompensator,
    direct_torque_output_scales_for_pyagxarm,
    get_default_model_path,
    get_midas_hand_model_path,
    get_payload_700g_model_path,
    gravity_vector_for_installation,
)
from scripts.piper_api.piper_api import mit_torque_limits_for_firmware  # type: ignore[import-not-found]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MuJoCo direct gravity compensation on a Piper arm."
    )
    parser.add_argument(
        "--model-path",
        default=str(get_midas_hand_model_path()),
        help=(
            "Path to a Piper MuJoCo XML model. Default: bundled "
            "assets/models/piper_grav_comp_midas_hand.xml"
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=("direct",),
        default="direct",
        help="Compensation model type. Only direct MuJoCo torque is supported locally.",
    )
    parser.add_argument(
        "--joint-names",
        nargs=6,
        default=list(DEFAULT_JOINT_NAMES),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Six MuJoCo joint names, in Piper joint order. Default: joint1 ... joint6.",
    )
    parser.add_argument(
        "--channel",
        "--can-port",
        dest="channel",
        default="can0",
        help="CAN channel/interface name. Default: can0.",
    )
    parser.add_argument(
        "--interface",
        default="socketcan",
        help="python-can interface. Default: socketcan.",
    )
    parser.add_argument(
        "--firmware",
        choices=("default", "v183", "v188"),
        default="v183",
        help=(
            "Piper firmware driver. Default: v183 (S-V1.8-3~7, MIT t_ff up to 8 N*m). "
            "Use v188 only for S-V1.8-8+ firmware."
        ),
    )
    parser.add_argument(
        "--installation-pos",
        choices=("horizontal", "left", "right", "unchanged"),
        default="right",
        help=(
            "Installation profile sent to Piper firmware and used to pick the "
            "MuJoCo gravity vector. Default: right (side mount, gravity along -Y)."
        ),
    )
    parser.add_argument(
        "--gravity",
        type=float,
        nargs=3,
        metavar=("GX", "GY", "GZ"),
        default=None,
        help=(
            "Override MuJoCo gravity vector in m/s^2. "
            "If omitted, derived from --installation-pos "
            "(right -> 0 -9.81 0, left -> 0 9.81 0, horizontal -> 0 0 -9.81)."
        ),
    )
    parser.add_argument(
        "--payload",
        choices=("empty", "half", "full", "unchanged"),
        default="unchanged",
        help="Firmware payload profile sent on connect. Default: unchanged.",
    )
    parser.add_argument(
        "--software-version",
        default=None,
        help=(
            "Override Piper software version for MIT torque scaling, e.g. S-V1.8-3. "
            "By default the script reads firmware over CAN after connect."
        ),
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=200.0,
        help="Compensation loop rate. Default: 200 (matches piper_control).",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.018,
        help=(
            "Viscous damping gain applied as -damping * joint_velocity. "
            "Default: 0.018 (matches piper_control)."
        ),
    )
    parser.add_argument(
        "--torque-scale",
        type=float,
        default=1.0,
        help="Global scale applied to all computed torque commands.",
    )
    parser.add_argument(
        "--coulomb-comp",
        type=float,
        nargs="+",
        default=None,
        metavar="NM",
        help=(
            "Coulomb friction compensation in N*m: torque added in the direction "
            "of motion (fc*sign(velocity)) to cancel joint friction and feel like "
            "teach/drag mode. Pass 1 value (all joints) or 6 (per joint). "
            "Default: 0 (off). Raise gradually; too high causes oscillation."
        ),
    )
    parser.add_argument(
        "--viscous-comp",
        type=float,
        nargs="+",
        default=None,
        metavar="NMS",
        help=(
            "Viscous friction compensation (fv*velocity, added with motion). "
            "Pass 1 or 6 values. Default: 0 (off)."
        ),
    )
    parser.add_argument(
        "--comp-velocity-deadband",
        type=float,
        default=0.02,
        help=(
            "Velocity (rad/s) below which Coulomb compensation is suppressed to "
            "avoid chatter at rest. Default: 0.02."
        ),
    )
    parser.add_argument(
        "--joint-torque-scale",
        type=float,
        nargs=6,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help=(
            "Per-joint multipliers applied after gravity/damping and before "
            "firmware output scaling. Default: 1 1 1 1 1 1. "
            "Avoid boosting J5/J6 unless you know the sign is correct."
        ),
    )
    parser.add_argument(
        "--mit-p-des",
        choices=("current", "zero"),
        default="current",
        help=(
            "MIT desired joint position reference. Default: current (use live "
            "joint angles). Use zero only if your firmware needs piper_sdk-style "
            "JointMitCtrl(0,0,0,0,torque) references."
        ),
    )
    parser.add_argument(
        "--hold-position",
        action="store_true",
        help=(
            "Track current joint angles with MIT kp/kd (stiff hold). "
            "Default is torque-only hover/drag mode (kp=0)."
        ),
    )
    parser.add_argument(
        "--mit-kp",
        type=float,
        default=0.0,
        help="MIT proportional gain. Default: 0 (torque-only / free motion).",
    )
    parser.add_argument(
        "--mit-kd",
        type=float,
        default=0.0,
        help="MIT derivative gain. Default: 0 (torque-only / free motion).",
    )
    parser.add_argument(
        "--ramp-seconds",
        type=float,
        default=2.0,
        help=(
            "Ramp torque-scale from 0 to the target over this many seconds. "
            "Default: 2.0. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-abs-torque",
        type=float,
        nargs=6,
        default=None,
        metavar=("T1", "T2", "T3", "T4", "T5", "T6"),
        help=(
            "Per-joint absolute torque limits in N*m. "
            "Defaults to driver limits (v183: 8, v188: 16, default: 32/8)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print computed commands without sending MIT torque commands.",
    )
    parser.add_argument(
        "--no-enable",
        action="store_true",
        help="Do not send enable before starting the loop.",
    )
    parser.add_argument(
        "--zero-velocity-if-missing",
        action="store_true",
        help="Use zero joint velocities if motor-state feedback is unavailable.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Stop after this many iterations. Default 0 runs until Ctrl-C.",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.5,
        help="Seconds between status prints. Default: 0.5.",
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Require pressing Enter before enabling the arm and sending torques.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=0.0,
        help=(
            "Stop automatically after this many seconds and send zero torques. "
            "Default: 0 (run until Ctrl-C)."
        ),
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help=(
            "Conservative hardware preset: low torque scale, no damping, long ramp, "
            "tight torque caps, confirm-live, and 30s max duration."
        ),
    )
    return parser.parse_args()


def apply_safe_preset(args) -> None:
    """Apply conservative defaults for first hardware bring-up."""
    args.torque_scale = min(args.torque_scale, 0.2)
    args.damping = 0.0
    args.ramp_seconds = max(args.ramp_seconds, 5.0)
    args.max_abs_torque = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]
    args.confirm_live = True
    if args.max_duration <= 0:
        args.max_duration = 30.0


def validate_args(args) -> None:
    if args.safe:
        apply_safe_preset(args)
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    if args.damping < 0:
        raise ValueError("--damping must be non-negative")
    if not math.isfinite(args.torque_scale) or args.torque_scale < 0:
        raise ValueError("--torque-scale must be finite and non-negative")
    joint_torque_scales = (
        list(args.joint_torque_scale)
        if args.joint_torque_scale is not None
        else [1.0] * 6
    )
    if any(scale < 0 or not math.isfinite(scale) for scale in joint_torque_scales):
        raise ValueError("--joint-torque-scale values must be finite and non-negative")
    args.joint_torque_scale = joint_torque_scales
    args.coulomb_comp = _normalize_comp(args.coulomb_comp, name="--coulomb-comp")
    args.viscous_comp = _normalize_comp(args.viscous_comp, name="--viscous-comp")
    if args.comp_velocity_deadband < 0 or not math.isfinite(args.comp_velocity_deadband):
        raise ValueError("--comp-velocity-deadband must be finite and non-negative")
    if args.hold_position and args.mit_kp == 0.0 and args.mit_kd == 0.0:
        args.mit_kp = 2.0
        args.mit_kd = 0.5
    if args.mit_kp < 0 or args.mit_kd < 0:
        raise ValueError("--mit-kp and --mit-kd must be non-negative")
    if not math.isfinite(args.mit_kp) or not math.isfinite(args.mit_kd):
        raise ValueError("--mit-kp and --mit-kd must be finite")
    if args.iterations < 0:
        raise ValueError("--iterations must be non-negative")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")
    if args.ramp_seconds < 0:
        raise ValueError("--ramp-seconds must be non-negative")
    if args.max_duration < 0:
        raise ValueError("--max-duration must be non-negative")
    torque_limits = (
        list(args.max_abs_torque)
        if args.max_abs_torque is not None
        else mit_torque_limits_for_firmware(args.firmware)
    )
    if any(limit <= 0 or not math.isfinite(limit) for limit in torque_limits):
        raise ValueError("--max-abs-torque values must be finite and positive")
    args.max_abs_torque = torque_limits


def _normalize_comp(values, *, name: str) -> list[float]:
    """Normalize a friction-comp arg (None/1/6 values) to a length-6 list."""
    if values is None:
        return [0.0] * 6
    if len(values) == 1:
        values = list(values) * 6
    if len(values) != 6:
        raise ValueError(f"{name} must be 1 or 6 values")
    if any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError(f"{name} values must be finite and non-negative")
    return list(values)


def clamp_torques(torques, limits):
    return [max(-limit, min(limit, torque)) for torque, limit in zip(torques, limits)]


def format_torques(label, torques):
    values = " ".join(f"{value:+7.3f}" for value in torques)
    return f"{label}: [{values}] N*m"


def read_live_state(api, *, zero_velocity_if_missing: bool):
    joint_state = api.get_joint_state()
    if joint_state is None:
        raise RuntimeError("Joint position feedback is unavailable")

    motor_state = api.get_motor_state()
    if motor_state is None:
        if not zero_velocity_if_missing:
            raise RuntimeError(
                "Motor-state feedback is unavailable; pass "
                "--zero-velocity-if-missing to compute without damping feedback"
            )
        velocities = [0.0] * 6
    else:
        velocities = motor_state.velocities

    return joint_state.joints, velocities, joint_state.timestamp


def resolve_software_version(api, override: str | None) -> str | None:
    if override:
        return override
    return api.get_software_version()


def run_loop(args) -> None:
    try:
        from scripts.piper_api.piper_api import PiperAPI
    except ModuleNotFoundError as exc:
        if exc.name == "can":
            raise RuntimeError(
                "python-can is required to talk to Piper. Install pyAgxArm's "
                "dependencies, for example `python3 -m pip install -e ../pyAgxArm` "
                "or `python3 -m pip install python-can`."
            ) from exc
        raise

    compensator = MujocoGravityCompensator(
        args.model_path,
        joint_names=args.joint_names,
        driver=args.firmware,
        gravity=args.gravity,
        installation_pos=None if args.gravity is not None else args.installation_pos,
    )
    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
        installation_pos=args.installation_pos,
        payload=args.payload,
    )

    period_s = 1.0 / args.rate_hz
    sent_live_torque = False
    software_version: str | None = None

    try:
        print(f"Loading model: {Path(args.model_path).expanduser()}")
        ok = api.connect()
        print(f"Piper connected: {'ok' if ok else 'not ok'}")
        software_version = resolve_software_version(api, args.software_version)
        compensator.set_firmware_version(
            software_version, driver=args.firmware
        )
        print(f"Software version: {software_version or 'unknown'}")
        print(f"MIT driver: {args.firmware}")
        print(f"Installation pos: {args.installation_pos}")
        gravity = compensator.get_gravity()
        print(
            "MuJoCo gravity (m/s^2): "
            f"{gravity[0]:+.2f} {gravity[1]:+.2f} {gravity[2]:+.2f}"
        )
        if args.gravity is None and args.installation_pos == "unchanged":
            print(
                "Warning: --installation-pos unchanged; MuJoCo gravity left at "
                "XML default (0 0 -9.81). Set --installation-pos or --gravity."
            )
        print(
            "MIT output scales (J1-J6): "
            + " ".join(f"{value:.2f}" for value in compensator.output_scales)
        )
        print(
            "Joint torque scales (J1-J6): "
            + " ".join(f"{value:.2f}" for value in args.joint_torque_scale)
        )
        if any(args.coulomb_comp) or any(args.viscous_comp):
            print(
                "Friction comp coulomb (N*m): "
                + " ".join(f"{v:.3f}" for v in args.coulomb_comp)
            )
            print(
                "Friction comp viscous (N*m*s): "
                + " ".join(f"{v:.3f}" for v in args.viscous_comp)
            )
        if software_version is None:
            print(
                "Warning: firmware version unavailable; using legacy MIT scaling "
                f"{direct_torque_output_scales_for_pyagxarm(None, driver=args.firmware)}"
            )

        joints, velocities, _ = read_live_state(
            api, zero_velocity_if_missing=args.zero_velocity_if_missing
        )
        print("Initial state OK.")
        print(
            f"Torque limits (N*m): {' '.join(f'{v:.1f}' for v in args.max_abs_torque)}"
        )
        if args.ramp_seconds > 0:
            print(
                f"Torque ramp: 0 -> {args.torque_scale:.3f} over "
                f"{args.ramp_seconds:.1f}s"
            )
        if args.max_duration > 0:
            print(f"Auto stop after {args.max_duration:.1f}s")
        if args.hold_position:
            print(
                f"Control mode: position hold "
                f"(p_des={args.mit_p_des}, kp={args.mit_kp:.2f}, kd={args.mit_kd:.2f})"
            )
        else:
            print(
                "Control mode: torque-only hover/drag "
                f"(p_des={args.mit_p_des}, kp={args.mit_kp:.2f}, kd={args.mit_kd:.2f})"
            )
        print("Dry run: no torque will be sent." if args.dry_run else "Live MIT torque output enabled.")

        if args.confirm_live and not args.dry_run:
            print(
                "\nSAFETY: Keep clear of the arm. Hold Ctrl-C ready to stop immediately."
            )
            input("Press Enter to enable MIT torque output (or Ctrl-C to abort)... ")

        if not args.no_enable and not args.dry_run:
            if not api.enable():
                raise RuntimeError("Timed out enabling Piper")
            api.set_mit_mode()
            print("MIT mode enabled.")

        count = 0
        loop_start_t = time.monotonic()
        next_print_t = loop_start_t
        next_tick_t = loop_start_t
        last_joint_timestamp: float | None = None
        stale_count = 0
        # At 200 Hz, 10 identical timestamps = ~50 ms of stale data.
        MAX_STALE_ITERS = 10
        while args.iterations == 0 or count < args.iterations:
            loop_t = time.monotonic()
            if args.max_duration > 0 and loop_t - loop_start_t >= args.max_duration:
                print(f"Reached --max-duration {args.max_duration:.1f}s; stopping.")
                break
            if args.ramp_seconds > 0:
                ramp = min(1.0, (loop_t - loop_start_t) / args.ramp_seconds)
            else:
                ramp = 1.0
            effective_torque_scale = args.torque_scale * ramp

            try:
                joints, velocities, joint_timestamp = read_live_state(
                    api, zero_velocity_if_missing=args.zero_velocity_if_missing
                )
            except RuntimeError:
                if not args.dry_run:
                    try:
                        api.move_mit_torques([0.0] * 6)
                    except Exception:
                        pass
                next_tick_t += period_s
                continue

            if joint_timestamp == last_joint_timestamp:
                stale_count += 1
                if stale_count >= MAX_STALE_ITERS:
                    if not args.dry_run:
                        try:
                            api.move_mit_torques([0.0] * 6)
                        except Exception:
                            pass
                    next_tick_t += period_s
                    continue
            else:
                stale_count = 0
                last_joint_timestamp = joint_timestamp

            result = compensator.compute(
                joints,
                joint_velocities=velocities,
                damping=args.damping,
                torque_scale=effective_torque_scale,
                joint_torque_scales=args.joint_torque_scale,
                coulomb_comp=args.coulomb_comp,
                viscous_comp=args.viscous_comp,
                comp_velocity_deadband=args.comp_velocity_deadband,
            )
            command_torques = clamp_torques(result.command_torques, args.max_abs_torque)

            if not args.dry_run:
                mit_positions = (
                    joints if args.mit_p_des == "current" else None
                )
                api.move_mit_torques(
                    command_torques,
                    p_des=mit_positions,
                    kp=args.mit_kp,
                    kd=args.mit_kd,
                )
                sent_live_torque = True

            if loop_t >= next_print_t:
                print(f"torque_scale: {effective_torque_scale:.3f}")
                print(format_torques("gravity", result.gravity_torques))
                print(format_torques("damping", result.damping_torques))
                print(format_torques("frictcmp", result.friction_comp_torques))
                print(format_torques("scaled", result.command_torques))
                next_print_t = loop_t + args.print_every

            count += 1
            next_tick_t += period_s
            sleep_s = next_tick_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick_t = time.monotonic()
    finally:
        try:
            if sent_live_torque:
                try:
                    api.move_mit_torques([0.0] * 6)
                    print("Sent zero MIT torques.")
                except Exception as exc:
                    print(f"Warning: could not send zero torques ({exc})")
        finally:
            api.disconnect()


def main():
    args = parse_args()
    validate_args(args)
    run_loop(args)


if __name__ == "__main__":
    main()
