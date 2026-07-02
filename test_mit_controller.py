#!/usr/bin/env python3
"""
MIT teacher drag 모드.

τ_cmd = g(q) + friction_comp(v)   (kp=0, kd=vel_damping)

손으로 끌면 따라오고, 놓으면 그 자리에 중력보상 상태로 정지.

Usage:
    python3 test_mit_controller.py --firmware v183 --coulomb-comp 0.3
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
    parser = argparse.ArgumentParser(description="MIT 임피던스 컨트롤러 테스트.")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--firmware", choices=("default", "v183", "v188"), default="default")
    parser.add_argument(
        "--installation-pos", choices=("horizontal", "left", "right"), default="right"
    )
    parser.add_argument("--payload", default="half")
    parser.add_argument("--rate-hz", type=float, default=120.0)
    parser.add_argument("--vel-damping", type=float, default=0.0,
                        metavar="B",
                        help="속도 비례 댐핑 kd (N·m·s/rad). 0=댐핑 없음. Default: 0.0")
    parser.add_argument("--coulomb-comp", type=float, default=0.0,
                        metavar="NM",
                        help="쿨롱 마찰 보상 (N·m). 운동 방향으로 토크 추가. Default: 0.0")
    parser.add_argument("--viscous-comp", type=float, default=0.0,
                        metavar="NMS",
                        help="점성 마찰 보상 (N·m·s/rad). 속도에 비례. Default: 0.0")
    parser.add_argument(
        "--model-path",
        default=str(Path(THIS_DIR) / "assets" / "models" / "piper_grav_comp_payload_700g.xml"),
    )
    parser.add_argument(
        "--gravity", type=float, nargs=3, metavar=("GX", "GY", "GZ"),
        default=[0.0, -9.81, 0.0],
        help="MuJoCo 중력 벡터 (m/s²). Default: right-mount (pure -Y).",
    )
    parser.add_argument("--torque-scale", type=float, default=1.2)
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
    print(f"Initial q (rad): {[f'{j:+.4f}' for j in joint_state.joints]}")

    api.enable()
    api.set_mit_mode()
    print(f"\nMIT @ {args.rate_hz:.0f} Hz  |  vel_damping={args.vel_damping}  coulomb={args.coulomb_comp}  viscous={args.viscous_comp}")
    print("Ctrl+C to stop.\n")

    period_s = 1.0 / args.rate_hz
    loop_count = 0
    last_report = time.monotonic()
    start_time = time.monotonic()

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

            grav_result = grav_comp.compute(
                q,
                joint_velocities=v,
                coulomb_comp=args.coulomb_comp,
                viscous_comp=args.viscous_comp,
                torque_scale=args.torque_scale,
            )

            api.move_mit_torques(
                grav_result.command_torques,
                p_des=q,
                v_des=[0.0] * 6,
                kp=0.0,
                kd=args.vel_damping,
                clamp=True,
            )
            loop_count += 1

            now = time.monotonic()
            if now - last_report >= 1.0:
                hz = loop_count / (now - last_report)
                last_report = now
                loop_count = 0
                print(
                    f"{hz:.1f} Hz | "
                    f"q: {[f'{j:+.3f}' for j in q]} | "
                    f"grav: {[f'{t:+.3f}' for t in grav_result.gravity_torques]}"
                )

            if (sleep := period_s - (time.monotonic() - t0)) > 0:
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
