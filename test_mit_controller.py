#!/usr/bin/env python3
"""
MIT impedance controller smoke-test with drag-and-hold mode.

두 가지 모드 (--mode):
  hold      : 초기 자세 유지. 밀면 복원력이 돌려보냄.
  drag-hold : 드래그 중 K=0, B만 끈적하게 저항. 놓으면 그 자리에 K 스프링 고정.

드래그 중 제어법칙:   τ = g(q)  +  0*(hold_q - q)  -  B*v
홀드 중 제어법칙:     τ = g(q)  +  K*(hold_q - q)  -  B_hold*v

Usage:
    python3 test_mit_controller.py --mode drag-hold --firmware v183
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from scripts.piper_api.gravity_compensation import MujocoGravityCompensator
from scripts.piper_api.piper_api import PiperAPI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIT impedance controller test.")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--firmware", choices=("default", "v183", "v188"), default="default")
    parser.add_argument(
        "--installation-pos", choices=("horizontal", "left", "right"), default="right"
    )
    parser.add_argument("--payload", default="half")
    parser.add_argument("--rate-hz", type=float, default=120.0)

    # ── Hold phase: K·(q_des - q) + B_hold·(0 - v) ──────────────────────────
    parser.add_argument("--hold-K", type=float, default=5.0,
                        metavar="K", help="Hold 스프링 강성 (N·m/rad). Default: 5.0")
    parser.add_argument("--hold-B", type=float, default=0.5,
                        metavar="B", help="Hold 댐핑 (N·m·s/rad). Default: 0.5")

    # ── Drag phase: 0·(q_des - q) + B_drag·(0 - v) ──────────────────────────
    parser.add_argument("--drag-B", type=float, default=0.5,
                        metavar="B", help="Drag 중 점성 댐핑 (N·m·s/rad). Default: 0.5")

    # ── Gravity comp ──────────────────────────────────────────────────────────
    parser.add_argument("--damping", type=float, default=0.018,
                        help="Gravity-comp 내 추가 댐핑. Default: 0.018")
    parser.add_argument(
        "--model-path",
        default=str(Path(THIS_DIR) / "assets" / "models" / "piper_grav_comp_payload_700g.xml"),
    )
    parser.add_argument(
        "--gravity", type=float, nargs=3, metavar=("GX", "GY", "GZ"),
        default=[0.240, -9.800, -0.375],
        help="MuJoCo 중력 벡터 (m/s²). Default: calibrated right-mount.",
    )
    parser.add_argument("--torque-scale", type=float, default=1.2)

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode", choices=("hold", "drag-hold"), default="hold",
    )
    parser.add_argument(
        "--drag-vel-threshold", type=float, default=0.03, metavar="RAD_S",
        help="이 속도(rad/s) 아래면 '멈춤' 판정. Default: 0.03",
    )
    parser.add_argument(
        "--drag-settle-cycles", type=int, default=8, metavar="N",
        help="N 사이클 연속 멈춤 후 hold로 전환. Default: 8 (~67ms @ 120Hz)",
    )
    parser.add_argument("--duration", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api = PiperAPI(
        firmware=args.firmware,
        interface=args.interface,
        channel=args.channel,
        installation_pos=args.installation_pos,
        payload=args.payload,
    )
    api.connect()
    if not api.is_ok():
        print("ERROR: Piper not responding on CAN.", file=sys.stderr)
        return 1
    print(f"Piper connected. receive FPS: {api.get_fps():.1f}")

    firmware_version = api.get_software_version()
    print(f"Firmware version: {firmware_version!r}")

    grav_comp = MujocoGravityCompensator(
        model_path=args.model_path,
        driver=args.firmware,
        firmware_version=firmware_version,
        installation_pos=args.installation_pos,
    )
    grav_comp.set_gravity(args.gravity)
    print(f"Gravity vector:  {grav_comp.get_gravity()}")
    print(f"Output scales:   {grav_comp.output_scales}")
    print(f"Torque scale:    {args.torque_scale}")

    joint_state = api.get_joint_state()
    if joint_state is None:
        print("ERROR: Could not read initial joint state.", file=sys.stderr)
        api.disconnect()
        return 1
    hold_q = list(joint_state.joints)
    print(f"Initial q (rad): {[f'{q:+.4f}' for q in hold_q]}")

    api.enable()
    api.set_mit_mode()

    drag_hold = args.mode == "drag-hold"
    if drag_hold:
        print(f"\nMIT drag-hold @ {args.rate_hz:.0f} Hz")
        print(f"  drag:  K=0,          B={args.drag_B}")
        print(f"  hold:  K={args.hold_K}, B={args.hold_B}")
    else:
        print(f"\nMIT hold @ {args.rate_hz:.0f} Hz")
        print(f"  K={args.hold_K}, B={args.hold_B}")
    print("Ctrl+C to stop.\n")

    period_s = 1.0 / args.rate_hz
    loop_count = 0
    last_report = time.monotonic()
    start_time = time.monotonic()

    settle_count = 0
    is_dragging = True  # 첫 사이클 충격 방지

    try:
        while True:
            t0 = time.monotonic()
            if args.duration is not None and (t0 - start_time) >= args.duration:
                break

            joint_state = api.get_joint_state()
            motor_state = api.get_motor_state()
            if joint_state is None:
                time.sleep(period_s)
                continue

            q = joint_state.joints
            v = motor_state.velocities if motor_state is not None else [0.0] * 6

            if drag_hold:
                max_v = max(abs(vi) for vi in v)
                if max_v > args.drag_vel_threshold:
                    hold_q = list(q)
                    settle_count = 0
                    is_dragging = True
                else:
                    settle_count += 1
                    if is_dragging and settle_count >= args.drag_settle_cycles:
                        hold_q = list(q)
                        is_dragging = False
                        print(f"Holding at: {[f'{p:+.4f}' for p in hold_q]}")

            grav_result = grav_comp.compute(
                q,
                joint_velocities=v,
                damping=args.damping,
                torque_scale=args.torque_scale,
            )

            if drag_hold and is_dragging:
                # K=0: p_des=q 로 스프링 제거, B_drag 만 남김
                api.move_mit_torques(
                    grav_result.command_torques,
                    p_des=q,
                    v_des=[0.0] * 6,
                    kp=0.0,
                    kd=args.drag_B,
                    clamp=True,
                )
            else:
                # K·(hold_q - q) + B_hold·(0 - v)
                api.move_mit_torques(
                    grav_result.command_torques,
                    p_des=hold_q,
                    v_des=[0.0] * 6,
                    kp=args.hold_K,
                    kd=args.hold_B,
                    clamp=True,
                )
            loop_count += 1

            now = time.monotonic()
            if now - last_report >= 1.0:
                hz = loop_count / (now - last_report)
                last_report = now
                loop_count = 0
                q_err = [q[i] - hold_q[i] for i in range(6)]
                state = "drag K=0" if (drag_hold and is_dragging) else f"hold K={args.hold_K}"
                print(
                    f"[{state}] {hz:.1f} Hz | "
                    f"q_err: {[f'{e:+.4f}' for e in q_err]} | "
                    f"grav: {[f'{t:+.3f}' for t in grav_result.gravity_torques]}"
                )

            elapsed = time.monotonic() - t0
            if (sleep := period_s - elapsed) > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nCtrl+C — stopping.")
    finally:
        api.disable()
        api.disconnect()
        print("Arm disabled and disconnected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
