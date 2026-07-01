#!/usr/bin/env python3
"""
MIT 임피던스 컨트롤러 위치 유지 / 스텝 응답 검증.

kp=0인 test_mit_controller.py와 달리, kp > 0에서 위치 제어가 제대로 작동하는지
(진동 없이 정확하게 유지하는지) 확인한다.

Usage:
    python3 test_mit_controller2.py                                      # 현재 위치 유지
    python3 test_mit_controller2.py --home                              # 홈 포즈로 이동 후 유지
    python3 test_mit_controller2.py --kp 5 --kp-wrist 10 --kd 0.5     # J4/J5/J6 kp=10
    python3 test_mit_controller2.py --ramp-time 3.0                    # kp 램프 시간 조정
    python3 test_mit_controller2.py --step-j 3 --step-rad 0.1         # J3을 0.1 rad 이동

    # Cartesian 이동 검증 (각 --eef-move 전후로 MIT hold 수행)
    python3 test_mit_controller2.py --kp 5 --kp-wrist 10 --kd 0.5 --torque-scale 1.05 --wrist-torque-scale 2.0 \\
        --eef-move 0 0 0.1 --eef-move 0 0.1 0 --hold-time 10
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from scripts.piper_api.gravity_compensation import MIDAS_HAND_MODEL_PATH, MujocoGravityCompensator
from scripts.piper_api.piper_api import PiperAPI

MIT_RATE_HZ = 120.0

# home 포즈 — test_curobo_reach.py 기준값
HOME_EEF = [0.325, -0.30, 0.20, -1.5707963, -0.659612, -1.5707963]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIT 임피던스 컨트롤러 위치 유지 검증")
    p.add_argument("--channel", default="can0")
    p.add_argument("--interface", default="socketcan")
    p.add_argument("--firmware", choices=("default", "v183", "v188"), default="v183")
    p.add_argument("--installation-pos", choices=("horizontal", "left", "right"), default="right")
    p.add_argument("--payload", default="half")
    p.add_argument(
        "--gravity", type=float, nargs=3, metavar=("GX", "GY", "GZ"),
        default=[0.240, -9.800, -0.375],
        help="MuJoCo 중력 벡터. Default: calibrated right-mount",
    )
    p.add_argument("--model-path", default=str(MIDAS_HAND_MODEL_PATH))
    p.add_argument("--torque-scale", type=float, default=1.05,
                   help="중력보상 토크 스케일 (J1-J6 공통). Default: 1.05")
    p.add_argument("--wrist-torque-scale", type=float, default=None,
                   help="J4/J5/J6 별도 torque 스케일. 미지정 시 --torque-scale과 동일. Default: None")
    p.add_argument("--kp", type=float, default=5.0,
                   help="위치 게인 J1-J3 (N·m/rad). Default: 5.0")
    p.add_argument("--kp-wrist", type=float, default=None,
                   help="J4/J5/J6 별도 위치 게인. 미지정 시 --kp와 동일. Default: None")
    p.add_argument("--kd", type=float, default=0.5,
                   help="속도 게인 (N·m·s/rad). Default: 0.5")
    p.add_argument("--hold-time", type=float, default=15.0,
                   help="총 유지 시간 (s). Default: 15.0")
    p.add_argument("--ramp-time", type=float, default=2.0,
                   help="MIT 진입 시 kp를 0.5→target으로 올리는 시간 (s). 0=즉시. Default: 2.0")

    # 홈 이동 옵션
    p.add_argument("--home", action="store_true",
                   help="MIT 모드 전에 firmware 위치제어로 홈 포즈 이동")
    p.add_argument("--home-settle", type=float, default=3.0,
                   help="홈 이동 후 정착 대기 시간 (s). Default: 3.0")

    # 스텝 응답 테스트
    p.add_argument("--step-j", type=int, choices=range(1, 7), metavar="JOINT",
                   help="스텝 인가할 조인트 (1-6)")
    p.add_argument("--step-rad", type=float, default=0.1,
                   help="스텝 크기 (rad). Default: 0.1")
    p.add_argument("--step-delay", type=float, default=5.0,
                   help="스텝 인가까지 대기 (s). Default: 5.0")

    # Cartesian 이동 시퀀스
    p.add_argument("--eef-move", type=float, nargs=3, action="append",
                   metavar=("DX", "DY", "DZ"),
                   help="firmware Cartesian 이동 (m). 여러 번 지정 가능. "
                        "각 이동 후 MIT hold 수행.")
    p.add_argument("--move-settle", type=float, default=4.0,
                   help="Cartesian 이동 후 정착 대기 (s). Default: 4.0")
    p.add_argument("--move-speed", type=int, default=30,
                   help="Cartesian 이동 속도 %% (1-100). Default: 30")

    return p.parse_args()


def _eef_str(eef: list[float]) -> str:
    xyz = [f"{v*100:+.1f}cm" for v in eef[:3]]
    rpy = [f"{math.degrees(v):+.1f}°" for v in eef[3:]]
    return f"xyz=[{', '.join(xyz)}]  rpy=[{', '.join(rpy)}]"


def wait_settle(api: PiperAPI, *, settle_s: float) -> None:
    """firmware 위치제어 후 속도가 충분히 작아질 때까지 대기."""
    print(f"  {settle_s:.1f}s 정착 대기...", end="", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < settle_s:
        ms = api.get_motor_state()
        if ms:
            max_vel = max(abs(v) for v in ms.velocities)
            if max_vel < 0.005:
                print(f" 정착 ({time.monotonic()-t0:.1f}s, max_vel={max_vel:.4f} rad/s)")
                return
        time.sleep(0.05)
    print(f" 완료 (타임아웃 {settle_s:.1f}s)")


def firmware_move_eef_delta(
    api: PiperAPI,
    delta_xyz: list[float],
    *,
    settle_s: float = 4.0,
    speed_percent: int = 30,
) -> list[float] | None:
    """현재 EEF에서 delta_xyz(m) 만큼 firmware Cartesian 제어로 이동.

    MIT 모드를 끊고 Position 모드로 전환 후 이동, settle 후 EEF 반환.
    """
    eef = api.get_eef_pose()
    if eef is None:
        print("  ERROR: EEF pose 읽기 실패, 이동 생략.")
        return None

    target = [eef[0] + delta_xyz[0],
              eef[1] + delta_xyz[1],
              eef[2] + delta_xyz[2],
              eef[3], eef[4], eef[5]]

    dx, dy, dz = delta_xyz
    print(f"\n  [Cartesian move] Δ=({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) m")
    print(f"    from: {_eef_str(eef)}")
    print(f"    to  : {_eef_str(target)}")

    # MIT → P 모드 전환: disable → enable → mode switch
    api.disable()
    time.sleep(0.2)
    api.enable()
    time.sleep(0.1)
    api.robot.set_motion_mode(api.robot.OPTIONS.MOTION_MODE.P)
    time.sleep(0.3)
    api.set_speed_percent(speed_percent)
    api.move_eef(target)
    wait_settle(api, settle_s=settle_s)

    eef_final = api.get_eef_pose()
    if eef_final:
        err = [eef_final[i] - target[i] for i in range(3)]
        print(f"    arrived: {_eef_str(eef_final)}")
        print(f"    EEF xyz error: ({err[0]*1000:+.1f}, {err[1]*1000:+.1f}, {err[2]*1000:+.1f}) mm")
    return eef_final


def _send_mit(
    api: PiperAPI,
    torques: list[float],
    p_des: list[float],
    v_des: list[float],
    kp_list: list[float],
    kd: float,
) -> None:
    """6-joint MIT command with per-joint kp."""
    for j in range(6):
        api.move_mit_torque(
            j + 1, torques[j],
            p_des=p_des[j], v_des=v_des[j],
            kp=kp_list[j], kd=kd, clamp=True,
        )


def _ramp_gains(
    api: PiperAPI,
    grav_comp: MujocoGravityCompensator,
    hold_pos: list[float],
    kp_target: float,
    kp_wrist: float,
    kd: float,
    torque_scale: float,
    ramp_time: float,
    joint_torque_scales: list[float] | None = None,
) -> None:
    """kp를 0.5 → kp_target으로 geomspace 램프 후 hold_pos 유지.

    J1-J3: kp_target, J4-J6: kp_wrist.
    """
    if ramp_time <= 0.0:
        return

    ramp_steps = max(2, int(round(ramp_time * MIT_RATE_HZ)))
    body_start = min(0.5, kp_target)
    wrist_start = min(0.5, kp_wrist)
    body_sched  = np.geomspace(body_start, kp_target, ramp_steps)
    wrist_sched = np.geomspace(wrist_start, kp_wrist, ramp_steps)

    period_ns = int(round(1_000_000_000 / MIT_RATE_HZ))
    next_ns = time.monotonic_ns()

    wrist_tag = f" | wrist {wrist_start:.2f}→{kp_wrist:.2f}" if kp_wrist != kp_target else ""
    print(
        f"  [ramp] kp {body_start:.2f}→{kp_target:.2f}{wrist_tag}"
        f"  ({ramp_time:.1f}s, {ramp_steps} steps) ...",
        end="", flush=True,
    )

    for body_kp, wrist_kp in zip(body_sched, wrist_sched):
        now_ns = time.monotonic_ns()
        if now_ns < next_ns:
            time.sleep(max(0.0, (next_ns - now_ns) / 1e9 - 0.0002))

        js = api.get_joint_state()
        ms = api.get_motor_state()
        if js is None:
            next_ns += period_ns
            continue

        q = js.joints
        v = ms.velocities if ms is not None else [0.0] * 6
        grav = grav_comp.compute(
            q, joint_velocities=v,
            torque_scale=torque_scale,
            joint_torque_scales=joint_torque_scales,
        )
        kp_list = [body_kp, body_kp, body_kp, wrist_kp, wrist_kp, wrist_kp]
        _send_mit(api, grav.command_torques, hold_pos, [0.0] * 6, kp_list, kd)

        next_ns += period_ns
        while next_ns < time.monotonic_ns():
            next_ns += period_ns

    print(" done")


def run_hold(
    api: PiperAPI,
    grav_comp: MujocoGravityCompensator,
    hold_time: float,
    kp: float,
    kp_wrist: float,
    kd: float,
    torque_scale: float,
    *,
    joint_torque_scales: list[float] | None = None,
    ramp_time: float = 2.0,
    step_joint: int | None = None,
    step_rad: float = 0.0,
    step_delay: float = 5.0,
) -> None:
    """MIT 위치 유지 루프. 매초 추적 오차를 출력한다."""
    period_ns = int(round(1_000_000_000 / MIT_RATE_HZ))
    next_ns = time.monotonic_ns()
    t_start = time.monotonic()

    hold_pos: list[float] | None = None
    step_applied = False

    # 초당 평균 오차 누적
    err_accum: list[list[float]] = [[] for _ in range(6)]
    last_report = time.monotonic()

    # 전체 RMS 계산용
    all_errors: list[np.ndarray] = []

    kp_tag = f"kp={kp}" + (f" / kp_wrist={kp_wrist}" if kp_wrist != kp else "")
    print(f"\n{'─'*74}")
    print(f"  MIT 위치 유지  {kp_tag}  kd={kd}  torque_scale={torque_scale}  ramp={ramp_time:.1f}s")
    print(f"  총 시간: {hold_time:.1f}s", end="")
    if step_joint:
        print(f"  |  {step_delay:.1f}s 후 J{step_joint} {step_rad:+.3f} rad 스텝", end="")
    print()
    print(f"{'─'*74}")

    # ── hold_pos 결정 → ramp ────────────────────────────────────────────────
    while hold_pos is None:
        js = api.get_joint_state()
        if js is not None:
            hold_pos = list(js.joints)
        else:
            time.sleep(0.01)
    print(f"  target q: {[f'{j:+.4f}' for j in hold_pos]}")
    _ramp_gains(api, grav_comp, hold_pos, kp, kp_wrist, kd, torque_scale, ramp_time,
                joint_torque_scales)

    # ── 헤더 ────────────────────────────────────────────────────────────────
    print(
        f"{'t(s)':>5}  "
        f"{'J1':>9}  {'J2':>9}  {'J3':>9}  "
        f"{'J4':>9}  {'J5':>9}  {'J6':>9}  "
        f"{'max|err|':>9}  {'step':>4}"
    )
    print(
        f"{'':>5}  "
        + ("  ".join(f"{'(mrad)':>9}") + "  ") * 6
        + f"{'(mrad)':>9}  {'':>4}"
    )
    print("─" * 74)

    next_ns = time.monotonic_ns()
    t_start = time.monotonic()

    try:
        while True:
            elapsed = time.monotonic() - t_start
            if elapsed >= hold_time:
                break

            now_ns = time.monotonic_ns()
            if now_ns < next_ns:
                time.sleep(max(0.0, (next_ns - now_ns) / 1e9 - 0.0002))
                continue

            js = api.get_joint_state()
            ms = api.get_motor_state()
            if js is None:
                next_ns += period_ns
                continue

            q = js.joints
            v = ms.velocities if ms is not None else [0.0] * 6
            # 스텝 인가
            if step_joint and not step_applied and elapsed >= step_delay:
                hold_pos[step_joint - 1] += step_rad
                step_applied = True
                print(f"\n  [STEP @ {elapsed:.2f}s] J{step_joint} → {hold_pos[step_joint-1]:+.4f} rad\n")

            grav = grav_comp.compute(
                q, joint_velocities=v,
                torque_scale=torque_scale,
                joint_torque_scales=joint_torque_scales,
            )
            kp_list = [kp, kp, kp, kp_wrist, kp_wrist, kp_wrist]
            _send_mit(api, grav.command_torques, hold_pos, [0.0] * 6, kp_list, kd)

            errs = [q[i] - hold_pos[i] for i in range(6)]
            for i in range(6):
                err_accum[i].append(errs[i])
            all_errors.append(np.array(errs))

            now = time.monotonic()
            if now - last_report >= 1.0:
                mean_abs = [float(np.mean(np.abs(err_accum[i]))) for i in range(6)]
                max_abs = max(mean_abs)
                print(
                    f"{elapsed:5.1f}  "
                    + "  ".join(f"{e*1000:+9.2f}" for e in mean_abs)
                    + f"  {max_abs*1000:9.2f}"
                    + f"  {'YES' if step_applied else 'no':>4}"
                )
                err_accum = [[] for _ in range(6)]
                last_report = now

            next_ns += period_ns
            while next_ns < time.monotonic_ns():
                next_ns += period_ns

    except KeyboardInterrupt:
        print("\nCtrl+C — 루프 중단.")

    # ── 전체 요약 ────────────────────────────────────────────────────────────
    if not all_errors:
        return

    arr = np.stack(all_errors, axis=0)   # (T, 6)
    rms  = np.sqrt(np.mean(arr ** 2, axis=0)) * 1000   # mrad
    peak = np.max(np.abs(arr), axis=0) * 1000           # mrad
    bias = np.mean(arr, axis=0) * 1000                  # mrad (drift direction)

    print(f"\n{'─'*74}")
    print("  요약 (전체 구간, 단위: mrad):")
    header2 = f"  {'':8}" + "".join(f"  {'J'+str(i+1):>9}" for i in range(6))
    print(header2)
    print(f"  {'RMS':8}" + "".join(f"  {rms[i]:+9.2f}" for i in range(6)))
    print(f"  {'Peak':8}" + "".join(f"  {peak[i]:+9.2f}" for i in range(6)))
    print(f"  {'Bias':8}" + "".join(f"  {bias[i]:+9.2f}" for i in range(6)))

    max_peak = float(np.max(peak))
    worst_j = int(np.argmax(peak)) + 1
    print()
    if max_peak < 5.0:
        verdict = "✓ 양호 (5 mrad 이내)"
    elif max_peak < 30.0:
        verdict = "△ 보통 (30 mrad 이내)"
    else:
        verdict = "✗ 불량 — 중력보상 부족 또는 kp 낮음"
    print(f"  최대 오차: {max_peak:.1f} mrad  (J{worst_j})  → {verdict}")

    # Bias 분석: 특정 조인트가 한쪽으로 계속 밀리면 중력 미보상
    max_bias_j = int(np.argmax(np.abs(bias))) + 1
    max_bias = float(bias[max_bias_j - 1])
    if abs(max_bias) > 10.0:
        direction = "양(+)" if max_bias > 0 else "음(-)"
        print(f"  편향: J{max_bias_j} {direction} 방향으로 {abs(max_bias):.1f} mrad "
              f"→ 해당 조인트 중력보상 부족 가능성")

    eef_final = api.get_eef_pose()
    if eef_final:
        print(f"\n  최종 EEF : {_eef_str(eef_final)}")


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
        print("ERROR: Piper CAN 응답 없음.", file=sys.stderr)
        return 1

    firmware_version = api.get_software_version()
    print(f"Piper connected.  firmware={firmware_version!r}  fps={api.get_fps():.1f}")

    js0 = api.get_joint_state()
    if js0 is None:
        print("ERROR: 초기 관절 상태 읽기 실패.", file=sys.stderr)
        api.disconnect()
        return 1
    eef0 = api.get_eef_pose()
    print(f"Initial q   : {[f'{j:+.4f}' for j in js0.joints]}")
    print(f"Initial EEF : {_eef_str(eef0) if eef0 else 'N/A'}")

    grav_comp = MujocoGravityCompensator(
        model_path=args.model_path,
        driver=args.firmware,
        firmware_version=firmware_version,
        installation_pos=args.installation_pos,
    )
    grav_comp.set_gravity(args.gravity)
    kp_wrist = args.kp_wrist if args.kp_wrist is not None else args.kp
    wts = args.wrist_torque_scale
    joint_torque_scales: list[float] | None = None
    if wts is not None and wts != args.torque_scale:
        joint_torque_scales = [1.0, 1.0, 1.0, wts / args.torque_scale,
                               wts / args.torque_scale, wts / args.torque_scale]
    print(f"Gravity      : {list(grav_comp.get_gravity())}")
    print(f"Output scales: {grav_comp.output_scales}")
    kp_tag = f"kp={args.kp}" + (f" / kp_wrist={kp_wrist}" if kp_wrist != args.kp else "")
    ts_tag = (f"torque_scale={args.torque_scale}"
              + (f" / wrist={wts}" if wts is not None and wts != args.torque_scale else ""))
    print(f"{kp_tag}  kd={args.kd}  {ts_tag}")

    eef_moves: list[list[float]] = args.eef_move or []

    # ── (선택) 홈 포즈 이동 ───────────────────────────────────────────────────
    if args.home:
        print(f"\n홈 포즈로 이동: {_eef_str(HOME_EEF)}")
        api.enable()
        time.sleep(0.1)
        api.robot.set_motion_mode(api.robot.OPTIONS.MOTION_MODE.P)
        time.sleep(0.3)
        api.move_eef(HOME_EEF)
        wait_settle(api, settle_s=args.home_settle)
        js_h = api.get_joint_state()
        eef_h = api.get_eef_pose()
        if js_h:
            print(f"  홈 q    : {[f'{j:+.4f}' for j in js_h.joints]}")
        if eef_h:
            print(f"  홈 EEF  : {_eef_str(eef_h)}")

    def _mit_hold(label: str) -> None:
        api.enable()
        api.set_mit_mode()
        print(f"\nMIT mode enabled. [{label}]")
        run_hold(
            api, grav_comp,
            hold_time=args.hold_time,
            kp=args.kp,
            kp_wrist=kp_wrist,
            kd=args.kd,
            torque_scale=args.torque_scale,
            joint_torque_scales=joint_torque_scales,
            ramp_time=args.ramp_time,
            step_joint=args.step_j if not eef_moves else None,
            step_rad=args.step_rad,
            step_delay=args.step_delay,
        )

    # ── MIT hold + Cartesian 이동 시퀀스 ─────────────────────────────────────
    total_segments = 1 + len(eef_moves)
    _mit_hold(f"1/{total_segments} 초기 위치")

    for seg_i, delta in enumerate(eef_moves, start=2):
        firmware_move_eef_delta(
            api, delta,
            settle_s=args.move_settle,
            speed_percent=args.move_speed,
        )
        _mit_hold(f"{seg_i}/{total_segments} 이동 후")

    api.disable()
    api.disconnect()
    print("Arm disabled and disconnected.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCtrl+C — 종료.")
        raise SystemExit(0)
