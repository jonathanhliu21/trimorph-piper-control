#!/usr/bin/env python3
"""cuRobo → MIT 제어 독립 검증 스크립트.

cuRobo가 지정한 EEF 목표점까지 계획을 세우고, MIT 임피던스 컨트롤러로 실행한 뒤
실제 도달 위치를 확인한다.

Usage:
    python3 test_curobo_reach.py
    python3 test_curobo_reach.py --firmware v183 --kp 30 --kd 2
    python3 test_curobo_reach.py --target 0.35 -0.25 0.25 -1.5708 -0.66 -1.5708
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from scripts.piper_api.gravity_compensation import (
    MIDAS_HAND_MODEL_PATH,
    MujocoGravityCompensator,
)
from scripts.piper_api.piper_api import PiperAPI


# ── 안전한 기본 테스트 waypoints (홈 포즈 주변) ──────────────────────────────
#   [x, y, z, roll, pitch, yaw]  (m / rad)  — 실제 로봇에서 검증된 값만 추가
DEFAULT_WAYPOINTS: list[list[float]] = [
    [0.325, -0.30, 0.20, -1.5707963, -0.659612, -1.5707963],  # home
    [0.325, -0.25, 0.25, -1.5707963, -0.659612, -1.5707963],  # home + 5cm up/side
    [0.325, -0.30, 0.20, -1.5707963, -0.659612, -1.5707963],  # home (돌아오기)
]

MIT_RATE_HZ = 120.0
HOLD_AFTER_REACH_S = 1.5   # 도달 후 이 시간만큼 유지하며 오차 측정


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cuRobo → MIT 단일 포인트 도달 테스트")
    p.add_argument("--channel", default="can0")
    p.add_argument("--interface", default="socketcan")
    p.add_argument("--firmware", choices=("default", "v183", "v188"), default="v183")
    p.add_argument("--installation-pos", default="right",
                   choices=("horizontal", "left", "right"))
    p.add_argument("--payload", default="half")
    p.add_argument(
        "--gravity", type=float, nargs=3, metavar=("GX", "GY", "GZ"),
        default=[0.240, -9.800, -0.375],
        help="MuJoCo 중력 벡터. Default: calibrated right-mount",
    )
    p.add_argument("--torque-scale", type=float, default=1.2)
    p.add_argument("--kp", type=float, default=30.0,
                   help="Position gain (N·m/rad). Default: 30.0")
    p.add_argument("--kd", type=float, default=2.0,
                   help="Velocity gain (N·m·s/rad). Default: 2.0")
    p.add_argument(
        "--target", type=float, nargs=6,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="단일 EEF 목표 [m, m, m, rad, rad, rad]. "
             "지정하면 DEFAULT_WAYPOINTS 대신 이 점만 실행.",
    )
    p.add_argument(
        "--urdf",
        default=str(THIS_DIR / "assets" / "models" / "piper_arm.urdf"),
        help="cuRobo용 URDF 경로",
    )
    p.add_argument("--timeout", type=float, default=10.0,
                   help="각 waypoint 실행 제한 시간 (s)")
    p.add_argument(
        "--model-path",
        default=str(MIDAS_HAND_MODEL_PATH),
        help="MuJoCo 중력보상 모델 경로",
    )
    p.add_argument("--no-wait", action="store_true",
                   help="waypoint 간 Enter 없이 자동 진행")
    return p.parse_args()


# ── Euler XYZ → 쿼터니언 [w, x, y, z] ──────────────────────────────────────
def _euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


# ── cuRobo v1 플래너 생성 ────────────────────────────────────────────────────
def _make_v1_planner(urdf_path: str):
    import torch
    from curobo.geom.sdf.world import WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState, RobotConfig
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    tensor_args = TensorDeviceType()
    robot_cfg = RobotConfig.from_basic(
        urdf_path=urdf_path,
        base_link="base_link",
        ee_link="flange_link",
        tensor_args=tensor_args,
    )
    # 바닥면 충돌체 (테이블)
    world_cfg = WorldConfig.from_dict({
        "cuboid": {
            "table": {
                "dims": [2.0, 2.0, 0.05],
                "pose": [0.0, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0],
            }
        }
    })
    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg, world_cfg, tensor_args,
        interpolation_dt=1.0 / MIT_RATE_HZ,
        use_cuda_graph=True,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
    )
    mg = MotionGen(mg_cfg)
    print("[cuRobo] warming up (JIT compile)...")
    mg.warmup()
    print("[cuRobo] warmup done.")

    plan_cfg = MotionGenPlanConfig(max_attempts=4, enable_graph=True, enable_opt=True)

    def plan(q_current: list[float], target_eef: list[float], reset: bool = False):
        x, y, z, roll, pitch, yaw = target_eef
        w, qx, qy, qz = _euler_xyz_to_quat(roll, pitch, yaw)
        pos_t = torch.tensor([[x, y, z]], dtype=torch.float32, device=tensor_args.device)
        quat_t = torch.tensor([[w, qx, qy, qz]], dtype=torch.float32, device=tensor_args.device)
        goal = Pose(position=pos_t, quaternion=quat_t)
        q_t = torch.tensor([q_current], dtype=torch.float32, device=tensor_args.device)
        start = JointState.from_position(q_t)
        result = mg.plan_single(start, goal, plan_cfg)
        if not result.success.item():
            return None
        return (
            result.interpolated_plan.position.cpu().numpy(),
            result.interpolated_plan.velocity.cpu().numpy(),
        )

    return plan, False


# ── cuRobo v2 MPC 플래너 생성 ────────────────────────────────────────────────
def _make_v2_mpc_planner(urdf_path: str, xrdf_path: str):
    """closed-loop MPC (curobo.model_predictive_control.ModelPredictiveControl).

    MotionGen/MotionPlanner와 달리 MPC는 목표가 그대로여도 매 tick 현재 관절
    상태를 피드백으로 다시 최적화해야 한다 (receding-horizon). 목표 포즈는
    바뀔 때만 갱신한다.
    """
    import torch
    from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
    from curobo.types import DeviceCfg, GoalToolPose, JointState
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
    from curobo._src.types.robot import RobotCfg

    device_cfg = DeviceCfg()
    kin_cfg = KinematicsCfg.from_robot_yaml_file(
        xrdf_path, tool_frames=["flange_link"],
        device_cfg=device_cfg, urdf_path=urdf_path,
    )
    robot_cfg = RobotCfg(kinematics=kin_cfg, device_cfg=device_cfg)
    cuboids = {"table": {"dims": [2.0, 2.0, 0.05], "pose": [0.0, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]}}
    interpolation_steps = 4  # cuRobo 고정값; command_dt = optimization_dt / interpolation_steps
    mpc_cfg = ModelPredictiveControlCfg.create(
        robot=robot_cfg,
        scene_model={"cuboid": cuboids},
        collision_cache={"cuboid": 1},
        device_cfg=device_cfg,
        self_collision_check=True,
        use_cuda_graph=True,
        optimization_dt=interpolation_steps / MIT_RATE_HZ,
        interpolation_steps=interpolation_steps,
    )
    mpc = ModelPredictiveControl(mpc_cfg)
    print("[cuRobo v2 MPC] warming up...")

    tool_frame = mpc.tool_frames[0]
    setup_done = False
    last_goal_eef: list[float] | None = None

    def plan(q_current: list[float], target_eef: list[float], reset: bool = False):
        nonlocal setup_done, last_goal_eef
        q_t = torch.tensor([q_current], dtype=device_cfg.dtype, device=device_cfg.device)
        current_state = JointState.from_position(q_t, joint_names=mpc.joint_names)

        if not setup_done:
            mpc.setup(current_state)
            setup_done = True
            last_goal_eef = None
            print("[cuRobo v2 MPC] warmup done.")
        elif reset:
            mpc.reset_robot(current_state)

        if target_eef != last_goal_eef:
            x, y, z, roll, pitch, yaw = target_eef
            w, qx, qy, qz = _euler_xyz_to_quat(roll, pitch, yaw)
            goal = GoalToolPose(
                tool_frames=[tool_frame],
                position=torch.tensor([[[[[x, y, z]]]]], dtype=device_cfg.dtype, device=device_cfg.device),
                quaternion=torch.tensor([[[[[w, qx, qy, qz]]]]], dtype=device_cfg.dtype, device=device_cfg.device),
            )
            # use_best_effort_ik: MPC의 내장 IK는 seed가 1개뿐이라 이미 도달
            # 가능한 포즈에서도 엄밀 수렴에 실패하는 경우가 잦다. MPC는 매
            # tick 목표를 향해 다시 최적화하므로 부정확한 IK 시드는 이후
            # tick에서 서서히 보정된다 (MotionGen처럼 한 번에 정확한 목표가
            # 필요하지 않음).
            if mpc.update_goal_tool_poses(goal, use_best_effort_ik=True):
                last_goal_eef = target_eef
            else:
                print(f"  [cuRobo v2 MPC] goal IK 실패: {target_eef}")

        result = mpc.optimize_action_sequence(current_state)
        if result.action_sequence is None:
            return None
        pos = result.action_sequence.position.detach()
        vel = result.action_sequence.velocity.detach()
        while pos.ndim > 2:
            pos, vel = pos[0], vel[0]
        return pos.cpu().numpy(), vel.cpu().numpy()

    return plan, True


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _make_planner(urdf_path: str):
    if _module_available("curobo.geom.sdf.world"):
        return _make_v1_planner(urdf_path)
    if _module_available("curobo.model_predictive_control"):
        xrdf = urdf_path.replace(".urdf", ".xrdf")
        return _make_v2_mpc_planner(urdf_path, xrdf)
    raise RuntimeError(
        "cuRobo v1 (MotionGen, curobo.geom.sdf.world) 또는 v2 MPC "
        "(curobo.model_predictive_control)가 필요합니다."
    )


# ── MIT 루프로 trajectory 실행 ───────────────────────────────────────────────
def execute_trajectory(
    api: PiperAPI,
    grav_comp: MujocoGravityCompensator,
    positions,        # (N, 6) ndarray
    velocities,       # (N, 6) ndarray
    kp: float,
    kd: float,
    torque_scale: float,
    timeout_s: float,
) -> bool:
    """trajectory를 MIT 120 Hz 루프로 실행. 완료 반환 True, 타임아웃 False."""
    period_ns = int(round(1_000_000_000 / MIT_RATE_HZ))
    n_waypoints = len(positions)
    idx = 0
    deadline = time.monotonic() + timeout_s
    next_ns = time.monotonic_ns()

    while idx < n_waypoints:
        if time.monotonic() > deadline:
            print(f"  [!] timeout  ({idx}/{n_waypoints} waypoints)")
            return False

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

        grav = grav_comp.compute(q, joint_velocities=v, torque_scale=torque_scale)

        # ── 실제 q에 가장 가까운 waypoint에서 시작 (최초 1회) ──────────────
        # (plan은 q_pre에서 시작하므로 실행 시점 q와 다를 수 있음)
        # idx==0일 때 한 번 seek 한다.
        if idx == 0:
            import numpy as np
            dists = np.linalg.norm(positions - np.array(q), axis=1)
            idx = int(np.argmin(dists))
            # 역방향 seek 방지: 앞의 waypoint만 허용
            idx = min(idx, n_waypoints // 4)

        p_des = positions[idx].tolist()
        v_des = velocities[idx].tolist()

        api.move_mit_torques(
            grav.command_torques,
            p_des=p_des,
            v_des=v_des,
            kp=kp,
            kd=kd,
            clamp=True,
        )
        idx += 1
        next_ns += period_ns
        while next_ns < time.monotonic_ns():
            next_ns += period_ns

    return True


def execute_mpc_reach(
    api: PiperAPI,
    grav_comp: MujocoGravityCompensator,
    plan,
    target_eef: list[float],
    kp: float,
    kd: float,
    torque_scale: float,
    timeout_s: float,
    pos_tol_m: float = 0.005,
    settle_s: float = 0.3,
) -> bool:
    """MPC를 매 tick 재호출해 목표에 도달할 때까지 실행 (closed-loop).

    MotionGen과 달리 MPC는 한 번의 plan()이 전체 궤적을 주지 않으므로
    (receding-horizon), 매 120Hz tick마다 현재 q로 다시 plan()을 호출하고
    그 결과의 마지막 waypoint를 MIT desired position/velocity로 사용한다.
    EEF 위치 오차가 pos_tol_m 이내로 settle_s동안 유지되면 성공.
    """
    period_ns = int(round(1_000_000_000 / MIT_RATE_HZ))
    deadline = time.monotonic() + timeout_s
    next_ns = time.monotonic_ns()
    converged_since: float | None = None

    while True:
        now = time.monotonic()
        if now > deadline:
            print("  [!] timeout")
            return False

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
        grav = grav_comp.compute(q, joint_velocities=v, torque_scale=torque_scale)

        planned = plan(q, target_eef)
        if planned is None:
            p_des, v_des = q, [0.0] * 6
        else:
            pos, vel = planned
            p_des, v_des = pos[-1].tolist(), vel[-1].tolist()

        api.move_mit_torques(
            grav.command_torques,
            p_des=p_des,
            v_des=v_des,
            kp=kp,
            kd=kd,
            clamp=True,
        )

        eef_now = api.get_eef_pose()
        if eef_now is not None:
            pos_err = math.sqrt(sum((eef_now[i] - target_eef[i]) ** 2 for i in range(3)))
            if pos_err < pos_tol_m:
                if converged_since is None:
                    converged_since = now
                elif now - converged_since >= settle_s:
                    return True
            else:
                converged_since = None

        next_ns += period_ns
        while next_ns < time.monotonic_ns():
            next_ns += period_ns


def hold_position(
    api: PiperAPI,
    grav_comp: MujocoGravityCompensator,
    hold_s: float,
    kp: float,
    kd: float,
    torque_scale: float,
) -> list[float] | None:
    """hold_s 동안 현재 위치 유지. 마지막 q 반환."""
    period_ns = int(round(1_000_000_000 / MIT_RATE_HZ))
    deadline = time.monotonic() + hold_s
    next_ns = time.monotonic_ns()
    hold_pos: list[float] | None = None
    last_q = None

    while time.monotonic() < deadline:
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
        last_q = q

        if hold_pos is None:
            hold_pos = list(q)

        grav = grav_comp.compute(q, joint_velocities=v, torque_scale=torque_scale)
        api.move_mit_torques(
            grav.command_torques,
            p_des=hold_pos,
            v_des=[0.0] * 6,
            kp=kp,
            kd=kd,
            clamp=True,
        )
        next_ns += period_ns
        while next_ns < time.monotonic_ns():
            next_ns += period_ns

    return last_q


def _eef_str(eef: list[float]) -> str:
    xyz = [f"{v*100:+.1f}cm" for v in eef[:3]]
    rpy = [f"{math.degrees(v):+.1f}°" for v in eef[3:]]
    return f"xyz=[{', '.join(xyz)}]  rpy=[{', '.join(rpy)}]"


def main() -> int:
    args = parse_args()

    waypoints = [args.target] if args.target else DEFAULT_WAYPOINTS
    print(f"Target waypoints ({len(waypoints)}):")
    for i, wp in enumerate(waypoints):
        print(f"  [{i}] {_eef_str(wp)}")
    print()

    # ── Piper 연결 ───────────────────────────────────────────────────────────
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

    # ── 중력보상 ─────────────────────────────────────────────────────────────
    grav_comp = MujocoGravityCompensator(
        model_path=args.model_path,
        driver=args.firmware,
        firmware_version=firmware_version,
        installation_pos=args.installation_pos,
    )
    grav_comp.set_gravity(args.gravity)
    print(f"Gravity      : {list(grav_comp.get_gravity())}")
    print(f"Output scales: {grav_comp.output_scales}")
    print(f"kp={args.kp}  kd={args.kd}  torque_scale={args.torque_scale}")

    # ── cuRobo 초기화 ────────────────────────────────────────────────────────
    try:
        plan, is_continuous = _make_planner(args.urdf)
    except Exception as exc:
        print(f"ERROR: cuRobo 초기화 실패: {exc}", file=sys.stderr)
        api.disconnect()
        return 1

    # ── MIT 모드 진입 ────────────────────────────────────────────────────────
    api.enable()
    api.set_mit_mode()
    print("\nMIT mode enabled. 0.5s 홀드...\n")
    hold_position(api, grav_comp, 0.5, args.kp, args.kd, args.torque_scale)

    # ── 각 waypoint 실행 ─────────────────────────────────────────────────────
    for wi, target_eef in enumerate(waypoints):
        print(f"\n{'='*60}")
        print(f"Waypoint {wi}: {_eef_str(target_eef)}")

        if not args.no_wait and wi > 0:
            input("  Enter 눌러서 실행 (Ctrl+C 종료): ")

        # 현재 관절 각도 읽기
        js = api.get_joint_state()
        if js is None:
            print("  ERROR: 관절 상태 읽기 실패, 건너뜀.")
            continue
        q_current = js.joints
        eef_now = api.get_eef_pose()
        print(f"  현재 EEF  : {_eef_str(eef_now) if eef_now else 'N/A'}")
        print(f"  현재 q    : {[f'{j:+.4f}' for j in q_current]}")
        print(f"  목표 EEF  : {_eef_str(target_eef)}")

        if is_continuous:
            # MPC: 계획+실행이 분리되지 않고 매 tick 재최적화하며 실행한다.
            print("  [cuRobo MPC] 실행 중 (closed-loop)...", end="", flush=True)
            t_exec = time.monotonic()
            ok = execute_mpc_reach(
                api, grav_comp, plan, target_eef,
                kp=args.kp, kd=args.kd,
                torque_scale=args.torque_scale,
                timeout_s=args.timeout,
            )
            t_exec = time.monotonic() - t_exec
            print(f" {'완료' if ok else 'TIMEOUT'}  ({t_exec*1000:.0f}ms)")
        else:
            # cuRobo 계획
            t_plan = time.monotonic()
            print("  [cuRobo] 계획 중...", end="", flush=True)
            planned = plan(q_current, target_eef)
            t_plan = time.monotonic() - t_plan

            if planned is None:
                print(f" FAILED ({t_plan*1000:.0f}ms)")
                print("  → IK 실패: 목표 포즈가 도달 불가능하거나 충돌 공간 내에 있음.")
                continue

            pos, vel = planned
            print(f" OK  ({t_plan*1000:.0f}ms, {len(pos)} waypoints, "
                  f"{len(pos)/MIT_RATE_HZ*1000:.0f}ms 소요 예상)")
            print(f"  traj start q: {[f'{v:+.4f}' for v in pos[0].tolist()]}")
            print(f"  traj end   q: {[f'{v:+.4f}' for v in pos[-1].tolist()]}")

            # 진단: 현재 q와 traj 시작 q 차이
            import numpy as np
            diff0 = np.array(q_current) - pos[0]
            print(f"  q_current vs pos[0] diff: {[f'{d:+.4f}' for d in diff0.tolist()]}")
            if np.linalg.norm(diff0) > 0.3:
                print("  [경고] traj 시작점과 현재 위치가 0.3 rad 이상 떨어져 있음!")

            # MIT 실행
            print("  실행 중...", end="", flush=True)
            t_exec = time.monotonic()
            ok = execute_trajectory(
                api, grav_comp, pos, vel,
                kp=args.kp, kd=args.kd,
                torque_scale=args.torque_scale,
                timeout_s=args.timeout,
            )
            t_exec = time.monotonic() - t_exec
            print(f" {'완료' if ok else 'TIMEOUT'}  ({t_exec*1000:.0f}ms)")

        # 홀드 + 오차 측정
        print(f"  {HOLD_AFTER_REACH_S:.1f}s 홀드 중...", end="", flush=True)
        q_final = hold_position(
            api, grav_comp, HOLD_AFTER_REACH_S,
            args.kp, args.kd, args.torque_scale,
        )
        print(" done")

        eef_final = api.get_eef_pose()
        if eef_final and q_final:
            print(f"  도달 EEF  : {_eef_str(eef_final)}")
            pos_err_mm = [abs(eef_final[i] - target_eef[i]) * 1000 for i in range(3)]
            ori_err_deg = [abs(math.degrees(eef_final[i] - target_eef[i])) for i in range(3, 6)]
            print(f"  위치 오차  : {[f'{e:.1f}mm' for e in pos_err_mm]}")
            print(f"  자세 오차  : {[f'{e:.1f}°'  for e in ori_err_deg]}")

    # ── 종료 ─────────────────────────────────────────────────────────────────
    print("\n모든 waypoint 완료. 1s 홀드 후 종료.")
    hold_position(api, grav_comp, 1.0, args.kp, args.kd, args.torque_scale)
    api.disconnect()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCtrl+C — 종료.")
        raise SystemExit(0)
