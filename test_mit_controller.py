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


class VelocityLowPassFilter:
    """Per-joint EMA filter for measured velocity fed into grav_comp.

    Raw firmware velocity is unfiltered, so feeding it straight into
    damping/friction-compensation (or a finite-difference acceleration
    estimate) turns encoder noise directly into torque noise/buzz.
    """

    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._state: list[float] | None = None

    def update(self, v: list[float]) -> list[float]:
        if self._alpha >= 1.0 or self._state is None:
            self._state = list(v)
        else:
            a = self._alpha
            self._state = [a * v[j] + (1.0 - a) * self._state[j] for j in range(6)]
        return self._state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIT 임피던스 컨트롤러 테스트.")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--firmware", choices=("default", "v183", "v188"), default="default")
    parser.add_argument(
        "--installation-pos", choices=("horizontal", "left", "right"), default="right"
    )
    parser.add_argument("--payload", default="half")
    parser.add_argument("--rate-hz", type=float, default=500.0,
                        help="제어 루프 목표 주파수(Hz). CAN 대역폭/펌웨어 피드백 속도에 따라 "
                             "실측치가 낮아질 수 있음 — 1초마다 출력되는 실측 Hz로 확인. "
                             "Default: 500.0 (기존 120.0에서 상향, 이 리포지토리에서 아직 검증된 적 "
                             "없는 값이므로 처음 실행 시 실측 Hz를 반드시 확인할 것).")
    parser.add_argument("--vel-damping", type=float, default=0.0,
                        metavar="B",
                        help="속도 비례 댐핑 kd (N·m·s/rad). 0=댐핑 없음. Default: 0.0")
    parser.add_argument("--coulomb-comp", type=float, default=0.0,
                        metavar="NM",
                        help="쿨롱 마찰 보상 (N·m). 운동 방향으로 토크 추가. Default: 0.0")
    parser.add_argument("--viscous-comp", type=float, default=0.0,
                        metavar="NMS",
                        help="점성 마찰 보상 (N·m·s/rad). 속도에 비례. Default: 0.0")
    parser.add_argument("--vel-filter-alpha", type=float, default=0.3,
                        metavar="A",
                        help="측정 속도 EMA 저역통과 필터 계수 (0,1]. 1.0=필터 없음. "
                             "댐핑/마찰보상 입력에만 적용. Default: 0.3 "
                             "(data_collection의 CUROBO_VEL_FILTER_ALPHA와 동일한 검증된 값).")
    parser.add_argument("--inertia-scale", type=float, default=0.0,
                        metavar="S",
                        help="관성 피드포워드 M(q)*qddot 스케일. 0.0=비활성화(기본값, opt-in). "
                             "가속도는 필터링된 속도의 유한차분 추정값이라 노이즈 증폭 위험 있음 — "
                             "0.1~0.3부터 시작 권장.")
    parser.add_argument("--accel-filter-alpha", type=float, default=0.15,
                        metavar="A",
                        help="유한차분 가속도에 추가로 거는 EMA 필터 계수 (0,1]. "
                             "속도 미분값은 노이즈가 더 크므로 --vel-filter-alpha보다 낮게(더 무겁게). "
                             "--inertia-scale=0.0이면 미사용. Default: 0.15.")
    parser.add_argument(
        "--model-path",
        default=str(Path(THIS_DIR) / "assets" / "models" / "midas_piper_grav_comp.xml"),
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
    print(
        f"\nMIT @ {args.rate_hz:.0f} Hz  |  vel_damping={args.vel_damping}  "
        f"coulomb={args.coulomb_comp}  viscous={args.viscous_comp}  "
        f"vel_filter_alpha={args.vel_filter_alpha}  inertia_scale={args.inertia_scale}"
    )
    print("Ctrl+C to stop.\n")

    period_s = 1.0 / args.rate_hz
    loop_count = 0
    last_report = time.monotonic()
    start_time = time.monotonic()
    vel_filter = VelocityLowPassFilter(args.vel_filter_alpha)
    accel_filter = VelocityLowPassFilter(args.accel_filter_alpha)
    v_prev: list[float] | None = None

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
            v_raw = motor_state.velocities if motor_state is not None else [0.0] * 6
            v = vel_filter.update(v_raw)

            if args.inertia_scale != 0.0 and v_prev is not None:
                qddot_raw = [(v[j] - v_prev[j]) / period_s for j in range(6)]
                qddot = accel_filter.update(qddot_raw)
            else:
                qddot = [0.0] * 6
            v_prev = list(v)

            grav_result = grav_comp.compute(
                q,
                joint_velocities=v,
                joint_accelerations=qddot,
                inertia_scale=args.inertia_scale,
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
                if hz < 0.8 * args.rate_hz:
                    print(
                        f"  WARNING: achieved {hz:.1f} Hz is <80% of requested "
                        f"{args.rate_hz:.0f} Hz — CAN/driver may be the bottleneck at this rate."
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
