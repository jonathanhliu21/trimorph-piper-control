#!/usr/bin/env python3
"""
MIT 임피던스 컨트롤러 위치 유지 / 스텝 응답 검증.

kp=0인 test_mit_controller.py와 달리, kp > 0에서 위치 제어가 제대로 작동하는지
(진동 없이 정확하게 유지하는지) 확인한다.

Usage:
    python3 test_mit_controller2.py                                      # 현재 위치 유지
    python3 test_mit_controller2.py --home                              # 홈 포즈로 이동 후 유지
    python3 test_mit_controller2.py --kp 5 --kp-wrist 20 --kd 0.5     # J4/J5/J6 kp=10
    python3 test_mit_controller2.py --ramp-time 3.0                    # kp 램프 시간 조정
    python3 test_mit_controller2.py --step-j 3 --step-rad 0.1         # J3을 0.1 rad 이동

    # Cartesian 이동 검증 (각 --eef-move 전후로 MIT hold 수행)
    # --eef-move는 firmware P 모드가 아니라 cuRobo로 계획한 조인트 궤적을
    # MIT 임피던스 루프(move_mit_torque)로 직접 실행한다.
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
from test_curobo_reach import (
    _make_planner,
    execute_mpc_reach_continuous,
    execute_mpc_reach_via_subgoals,
    execute_trajectory,
)

MIT_RATE_HZ = 120.0

# home 포즈 — test_curobo_reach.py 기준값
HOME_EEF = [0.325, -0.30, 0.20, -1.5707963, -0.659612, -1.5707963]

# room 좌표계(DX=앞, DY=오른쪽, DZ=수직 위 — 사람이 로봇을 보는 기준) → base
# frame 변환. get_eef_pose()/move_eef()/cuRobo는 모두 base_link 고정 좌표계를
# 쓰는데, installation_pos에 따라 이 base 축들이 room에서 어느 방향을
# 가리키는지가 달라진다.
#
# right 설치: base X/Y/Z 축을 하나씩 순수하게 움직여 실측 검증함
#   (--eef-move-frame base로 0.05 0 0 / 0 0.05 0 / 0 0 0.05 각각 실행):
#     base +X → 앞   base +Y → 위   base +Z → 오른쪽
#   (이전 버전은 gravity_compensation의 중력벡터 테이블만 보고 추정한 것으로,
#   중력 방향(2자유도)만으로는 중력축 기준 나머지 회전(1자유도)을 못 정하기
#   때문에 틀렸었다 — 실측으로 정정.)
#
# left 설치: 아직 실측 검증 안 됨 (미검증 추정치, 주의해서 사용).
def _room_delta_to_base(delta_room: list[float], installation_pos: str) -> list[float]:
    dx, dy, dz = delta_room  # room: dx=앞, dy=오른쪽, dz=위
    key = installation_pos.strip().lower()
    if key == "right":
        return [dx, dz, dy]  # base_X=앞(dx)  base_Y=위(dz)  base_Z=오른쪽(dy)
    if key == "left":
        print("  [!] left 설치의 room→base 매핑은 실측 검증되지 않았습니다 "
              "(추정치). --eef-move-frame base로 축별 실측을 먼저 해보세요.")
        return [dx, -dz, dy]
    return [dx, dy, dz]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIT 임피던스 컨트롤러 위치 유지 검증")
    p.add_argument("--channel", default="can0")
    p.add_argument("--interface", default="socketcan")
    p.add_argument("--firmware", choices=("default", "v183", "v188"), default="v183")
    p.add_argument("--installation-pos", choices=("horizontal", "left", "right"), default="right")
    p.add_argument("--payload", default="half")
    p.add_argument(
        "--gravity", type=float, nargs=3, metavar=("GX", "GY", "GZ"),
        default=[0.0, -9.81, 0.0],
        help="MuJoCo 중력 벡터. Default: right-mount (pure -Y)",
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

    # Cartesian 이동 시퀀스 (cuRobo 계획 + MIT 실행)
    p.add_argument("--eef-move", type=float, nargs=3, action="append",
                   metavar=("DX", "DY", "DZ"),
                   help="cuRobo로 계획한 조인트 궤적을 MIT 루프로 실행하는 Cartesian 이동 (m). "
                        "--eef-move-frame으로 room/base 좌표계 선택. "
                        "여러 번 지정 가능. 각 이동 후 MIT hold 수행.")
    p.add_argument("--eef-move-frame", choices=("room", "base"), default="room",
                   help="--eef-move 델타의 기준 좌표계. 'room'(기본값)은 사람이 로봇을 보는 기준 "
                        "(DX=앞, DY=오른쪽, DZ=수직 위)으로 받아 --installation-pos에 따라 base "
                        "frame으로 자동 변환 (right 설치만 실측 검증됨). "
                        "'base'는 변환 없이 base_link 축(X/Y/Z)을 그대로 사용 — 축별 실측 캘리브레이션용.")
    p.add_argument("--eef-move-continuous", action="store_true",
                   help="정적 sub-goal 대신, 매 tick 목표를 현재 위치에서 최종 목표 쪽으로 "
                        "--carrot-speed-cm-s만큼 전진시키는 '움직이는 당근' 방식으로 실행 "
                        "(실제 teleop의 연속 목표 갱신 패턴을 재현 — drift 재현/디버깅용)")
    p.add_argument("--carrot-speed-cm-s", type=float, default=5.0,
                   help="--eef-move-continuous 사용 시 목표가 전진하는 속도 (cm/s). Default: 5.0")
    p.add_argument("--carrot-goal-threshold-mm", type=float, default=5.0,
                   help="당근이 이만큼(mm) 전진했을 때만 실제 plan() 목표를 갱신 — 매 tick 목표가 "
                        "바뀌면 IK 시드가 정착할 시간이 없어 팔이 멈추는 현상과 연관된 것으로 "
                        "의심됨. Default: 5.0")
    p.add_argument(
        "--urdf",
        default=str(THIS_DIR / "assets" / "models" / "piper_arm.urdf"),
        help="cuRobo용 URDF 경로 (--eef-move 사용 시에만 로드)",
    )
    p.add_argument("--eef-move-timeout", type=float, default=40.0,
                   help="--eef-move 1회 실행 제한 시간 (s). Default: 40.0")
    p.add_argument("--subgoal-step-m", type=float, default=0.03,
                   help="cuRobo MPC sub-goal 간 최대 위치 간격 (m). Default: 0.03")
    p.add_argument("--subgoal-step-deg", type=float, default=5.0,
                   help="cuRobo MPC sub-goal 간 최대 자세 간격 (deg). Default: 5.0")
    p.add_argument("--subgoal-timeout", type=float, default=6.0,
                   help="cuRobo MPC sub-goal 하나당 제한 시간 (s). Default: 6.0")
    p.add_argument("--mpc-no-decel", action="store_true",
                   help="v2 MPC의 내장 실패시 감속(use_deceleration_on_failure)을 끈다 (진단용, 기본 비활성화 안 함)")
    p.add_argument("--mpc-no-cuda-graph", action="store_true",
                   help="v2 MPC의 CUDA graph capture(use_cuda_graph)를 끄고 매 tick eager 모드로 계산 "
                        "(수학은 동일, 실행만 느려짐 — graph 캡처/재생 자체가 원인인지 격리용)")
    p.add_argument("--mpc-accel-reg-scale", type=float, default=1.0,
                   help="v2 MPC lbfgs_mpc.yml의 가속도 regularization weight(기본 10000)에 곱하는 배율. "
                        "1.0보다 작게 주면 목표까지 더 빨리 갈 수 있지만 저크/진동이 늘 수 있음 (진단용). Default: 1.0")
    p.add_argument("--mpc-rate-hz", type=float, default=None,
                   help="v2 MPC가 내부적으로 가정하는 tick 간 경과 시간(optimization_dt=4/이값)을 정한다. "
                        "기본은 MIT_RATE_HZ(120)와 동일 — GPU가 느려 실제 tick rate가 이보다 낮으면 "
                        "(예: 60) MPC의 속도/가속도 적분 가정이 실제와 어긋난다. 실측 rate로 맞추려면 지정.")
    p.add_argument("--mpc-no-scene-collision", action="store_true",
                   help="v2 MPC의 가상 table cuboid 충돌체를 끈다. horizontal(Z=0이 바닥) 가정으로 넣은 "
                        "물체라 right/left 설치에서는 잘못된 위치에 있을 수 있음 (진단용)")
    p.add_argument("--mpc-no-self-collision", action="store_true",
                   help="v2 MPC의 self-collision 체크를 끈다 (진단용)")
    p.add_argument("--mpc-no-idle-tick", action="store_true",
                   help="MIT hold 구간에서 MPC를 계속 '데워두는' idle-tick(제자리 목표로 plan() 반복 호출)을 끈다. "
                        "거리=0에 가까운 문제를 수천 tick 반복하면 LBFGS warm-start history가 오염돼 "
                        "다음 진짜 이동의 첫 sub-goal에 오히려 방해가 될 수 있음 (진단용)")
    p.add_argument("--mpc-debug", action="store_true",
                   help="v2 MPC의 매 30 tick마다 옵티마이저 내부 상태(position_error, success, "
                        "feasible, 계획된 최대 관절속도)를 출력한다 (진단용)")

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


def mit_move_to_eef(
    api: PiperAPI,
    grav_comp: MujocoGravityCompensator,
    plan,
    is_continuous: bool,
    target_eef: list[float],
    *,
    kp: float,
    kp_wrist: float,
    kd: float,
    torque_scale: float,
    joint_torque_scales: list[float] | None,
    timeout_s: float,
    subgoal_step_m: float,
    subgoal_step_rad: float,
    subgoal_timeout_s: float,
    continuous_carrot: bool = False,
    carrot_speed_m_s: float = 0.05,
    carrot_goal_threshold_m: float = 0.005,
) -> list[float] | None:
    """target_eef(절대 좌표)까지, cuRobo가 계획한 조인트 궤적을 MIT 루프로 실행해 이동.

    firmware P 모드로 전환하지 않는다 — MIT 모드를 유지한 채 목표 조인트
    위치/속도(p_des/v_des)를 move_mit_torque로 스트리밍한다.
    """
    eef = api.get_eef_pose()
    js = api.get_joint_state()
    if eef is None or js is None:
        print("  ERROR: EEF/joint 상태 읽기 실패, 이동 생략.")
        return None

    print(f"\n  [MIT Cartesian move]")
    print(f"    from: {_eef_str(eef)}")
    print(f"    to  : {_eef_str(target_eef)}")

    if is_continuous and continuous_carrot:
        ok = execute_mpc_reach_continuous(
            api, grav_comp, plan, target_eef,
            kp=kp, kp_wrist=kp_wrist, kd=kd,
            torque_scale=torque_scale,
            timeout_s=timeout_s,
            joint_torque_scales=joint_torque_scales,
            carrot_speed_m_s=carrot_speed_m_s,
            goal_update_threshold_m=carrot_goal_threshold_m,
        )
    elif is_continuous:
        ok = execute_mpc_reach_via_subgoals(
            api, grav_comp, plan, target_eef,
            kp=kp, kp_wrist=kp_wrist, kd=kd,
            torque_scale=torque_scale,
            timeout_s=timeout_s,
            joint_torque_scales=joint_torque_scales,
            subgoal_step_m=subgoal_step_m,
            subgoal_step_rad=subgoal_step_rad,
            subgoal_timeout_s=subgoal_timeout_s,
        )
    else:
        planned = plan(list(js.joints), target_eef)
        if planned is None:
            print("  ERROR: cuRobo 계획 실패 (IK 불가 또는 충돌), 이동 생략.")
            return api.get_eef_pose()
        pos, vel = planned
        print(f"    traj: {len(pos)} waypoints (~{len(pos) / MIT_RATE_HZ * 1000:.0f}ms)")
        ok = execute_trajectory(
            api, grav_comp, pos, vel,
            kp=kp, kp_wrist=kp_wrist, kd=kd,
            torque_scale=torque_scale,
            timeout_s=timeout_s,
            joint_torque_scales=joint_torque_scales,
        )
    print(f"    {'완료' if ok else 'TIMEOUT/실패'}")

    eef_final = api.get_eef_pose()
    if eef_final:
        err = [eef_final[i] - target_eef[i] for i in range(3)]
        print(f"    arrived: {_eef_str(eef_final)}")
        print(f"    EEF xyz error: ({err[0]*1000:+.1f}, {err[1]*1000:+.1f}, {err[2]*1000:+.1f}) mm")
    return eef_final


def mit_move_eef_delta(
    api: PiperAPI,
    grav_comp: MujocoGravityCompensator,
    plan,
    is_continuous: bool,
    delta_xyz: list[float],
    **kwargs,
) -> list[float] | None:
    """현재 EEF에서 delta_xyz(m) 만큼 MIT 루프로 이동 (mit_move_to_eef 참고)."""
    eef = api.get_eef_pose()
    if eef is None:
        print("  ERROR: EEF pose 읽기 실패, 이동 생략.")
        return None
    target_eef = [eef[0] + delta_xyz[0],
                  eef[1] + delta_xyz[1],
                  eef[2] + delta_xyz[2],
                  eef[3], eef[4], eef[5]]
    dx, dy, dz = delta_xyz
    print(f"  Δ=({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) m")
    return mit_move_to_eef(api, grav_comp, plan, is_continuous, target_eef, **kwargs)


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
    curobo_plan=None,
    curobo_hold_eef: list[float] | None = None,
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

        if curobo_plan is not None and curobo_hold_eef is not None:
            curobo_plan(q, curobo_hold_eef)

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
    curobo_plan=None,
) -> None:
    """MIT 위치 유지 루프. 매초 추적 오차를 출력한다.

    curobo_plan이 주어지면 (v2 MPC), hold 구간 내내 현재 위치를 goal로
    매 tick plan()을 계속 호출해 MPC를 '데워진' 상태로 유지한다 (결과는
    버림). 그렇지 않으면 hold 중 cuRobo가 완전히 idle 상태가 되어, hold
    직후 첫 실제 Cartesian 이동의 sub-goal이 warm-start 없이 처음부터
    다시 수렴해야 해서 거의 진행하지 못하는 현상이 관측됐다.
    """
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
    hold_eef = api.get_eef_pose() if curobo_plan is not None else None
    _ramp_gains(api, grav_comp, hold_pos, kp, kp_wrist, kd, torque_scale, ramp_time,
                joint_torque_scales, curobo_plan=curobo_plan, curobo_hold_eef=hold_eef)

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

            if curobo_plan is not None and hold_eef is not None:
                curobo_plan(q, hold_eef)

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

    # ── (선택) cuRobo 플래너 초기화 (--eef-move 사용 시에만) ────────────────────
    plan = None
    is_continuous = False
    if eef_moves:
        try:
            plan, is_continuous = _make_planner(
                args.urdf,
                use_deceleration_on_failure=not args.mpc_no_decel,
                use_cuda_graph=not args.mpc_no_cuda_graph,
                accel_regularization_scale=args.mpc_accel_reg_scale,
                mpc_rate_hz=args.mpc_rate_hz if args.mpc_rate_hz is not None else MIT_RATE_HZ,
                enable_scene_collision=not args.mpc_no_scene_collision,
                enable_self_collision=not args.mpc_no_self_collision,
                debug=args.mpc_debug,
            )
        except Exception as exc:
            print(f"ERROR: cuRobo 초기화 실패: {exc}", file=sys.stderr)
            api.disconnect()
            return 1

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
            curobo_plan=plan if (eef_moves and is_continuous and not args.mpc_no_idle_tick) else None,
        )

    mit_move_kwargs = dict(
        kp=args.kp,
        kp_wrist=kp_wrist,
        kd=args.kd,
        torque_scale=args.torque_scale,
        joint_torque_scales=joint_torque_scales,
        timeout_s=args.eef_move_timeout,
        subgoal_step_m=args.subgoal_step_m,
        subgoal_step_rad=math.radians(args.subgoal_step_deg),
        subgoal_timeout_s=args.subgoal_timeout,
        continuous_carrot=args.eef_move_continuous,
        carrot_speed_m_s=args.carrot_speed_cm_s / 100.0,
        carrot_goal_threshold_m=args.carrot_goal_threshold_mm / 1000.0,
    )

    # ── cuRobo v2 MPC 사전 워밍업 ────────────────────────────────────────────
    # mpc.setup()(CUDA graph capture 등 무거운 1회성 초기화)은 plan()의 첫
    # 실제 호출 시점에 지연 실행된다. 여기서 미리 트리거해두지 않으면 그
    # 비용이 첫 --eef-move의 timeout 예산을 통째로 먹어 첫 이동이 0% 진행률로
    # 타임아웃난다 (실측됨). 목표를 현재 위치로 줘서 실제 이동은 발생시키지
    # 않는다.
    if eef_moves and is_continuous:
        js_warm = api.get_joint_state()
        eef_warm = api.get_eef_pose()
        if js_warm is not None and eef_warm is not None:
            print("[cuRobo v2 MPC] 사전 워밍업 중...", end="", flush=True)
            t_warm = time.monotonic()
            plan(list(js_warm.joints), eef_warm, reset=True)
            print(f" done ({time.monotonic()-t_warm:.1f}s)")

    # ── MIT hold + Cartesian 이동 시퀀스 ─────────────────────────────────────
    total_segments = 1 + len(eef_moves)
    _mit_hold(f"1/{total_segments} 초기 위치")

    for seg_i, delta_in in enumerate(eef_moves, start=2):
        if args.eef_move_frame == "base":
            delta_base = list(delta_in)
            print(f"  [base] Δ=({delta_base[0]:+.3f}, {delta_base[1]:+.3f}, {delta_base[2]:+.3f})  (변환 없음)")
        else:
            delta_base = _room_delta_to_base(delta_in, args.installation_pos)
            if delta_base != delta_in:
                print(f"  [room→base] room Δ=({delta_in[0]:+.3f}, {delta_in[1]:+.3f}, {delta_in[2]:+.3f})  "
                      f"→ base Δ=({delta_base[0]:+.3f}, {delta_base[1]:+.3f}, {delta_base[2]:+.3f})")
        mit_move_eef_delta(api, grav_comp, plan, is_continuous, delta_base, **mit_move_kwargs)
        _mit_hold(f"{seg_i}/{total_segments} 이동 후")

    # ── 종료 전 홈 복귀 (--eef-move로 이동한 채 disable하면 낙하하므로) ──────────
    if eef_moves:
        print(f"\n홈 포즈로 복귀: {_eef_str(HOME_EEF)}")
        mit_move_to_eef(api, grav_comp, plan, is_continuous, HOME_EEF, **mit_move_kwargs)
        _mit_hold("홈 복귀 후")

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
