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

import numpy as np

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
        default=[0.0, -9.81, 0.0],
        help="MuJoCo 중력 벡터. Default: right-mount (pure -Y)",
    )
    p.add_argument("--torque-scale", type=float, default=1.2)
    p.add_argument("--wrist-torque-scale", type=float, default=None,
                   help="J4/J5/J6 별도 torque 스케일. 미지정 시 --torque-scale과 동일. Default: None")
    p.add_argument("--kp", type=float, default=30.0,
                   help="Position gain (N·m/rad). Default: 30.0")
    p.add_argument("--kp-wrist", type=float, default=None,
                   help="J4/J5/J6 별도 위치 게인. 미지정 시 --kp와 동일. Default: None")
    p.add_argument("--kd", type=float, default=2.0,
                   help="Velocity gain (N·m·s/rad). Default: 2.0")
    p.add_argument("--ramp-time", type=float, default=2.0,
                   help="MIT 진입 시 kp를 0.5→target으로 올리는 시간 (s). 0=즉시. Default: 2.0")
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
    p.add_argument("--timeout", type=float, default=40.0,
                   help="각 waypoint 실행 제한 시간 (s, 모든 sub-goal 합산). Default: 40.0")
    p.add_argument(
        "--model-path",
        default=str(MIDAS_HAND_MODEL_PATH),
        help="MuJoCo 중력보상 모델 경로",
    )
    p.add_argument("--no-wait", action="store_true",
                   help="waypoint 간 Enter 없이 자동 진행")
    p.add_argument("--subgoal-step-m", type=float, default=0.03,
                   help="MPC sub-goal 간 최대 위치 간격 (m). Default: 0.03")
    p.add_argument("--subgoal-step-deg", type=float, default=5.0,
                   help="MPC sub-goal 간 최대 자세 간격 (deg). Default: 5.0")
    p.add_argument("--subgoal-timeout", type=float, default=6.0,
                   help="sub-goal 하나당 제한 시간 (s). Default: 6.0")
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
    p.add_argument("--mpc-debug", action="store_true",
                   help="v2 MPC의 매 30 tick마다 옵티마이저 내부 상태(position_error, success, "
                        "feasible, 계획된 최대 관절속도)를 출력한다 (진단용)")
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
def _make_v2_mpc_planner(
    urdf_path: str, xrdf_path: str,
    use_deceleration_on_failure: bool = True,
    use_cuda_graph: bool = True,
    accel_regularization_scale: float = 1.0,
    mpc_rate_hz: float = MIT_RATE_HZ,
    enable_scene_collision: bool = True,
    enable_self_collision: bool = True,
    debug: bool = False,
):
    """closed-loop MPC (curobo.model_predictive_control.ModelPredictiveControl).

    MotionGen/MotionPlanner와 달리 MPC는 목표가 그대로여도 매 tick 현재 관절
    상태를 피드백으로 다시 최적화해야 한다 (receding-horizon). 목표 포즈는
    바뀔 때만 갱신한다.

    use_deceleration_on_failure=True(cuRobo 기본값)이면, 옵티마이저가 자기
    해를 신뢰할 수 없다고 판단할 때마다 최대 2초짜리 지수 감속 프로파일로
    스스로 브레이크를 건다. 실측 하드웨어에서 kp/kd나 중력보상 잔차 오차와
    맞물려 이 감속이 반복 트리거되면 "조금 가고 멈칫" 하는 크리핑으로
    보일 수 있다 — False로 끄면 이 내장 안전 감속이 비활성화된다 (진단용).

    accel_regularization_scale은 mpc/lbfgs_mpc.yml의
    rollout.cost_cfg.cspace_cfg.squared_l2_regularization_weight 중 가속도
    항목(기본 10000, 매우 큼)에 곱하는 배율이다. 1.0보다 작게 주면 "부드럽게
    움직이기"에 대한 페널티가 줄어 목표까지 더 빨리 갈 수 있지만, 대신
    저크/진동이 늘어날 수 있다 (진단용 — kp/decel/cuda-graph를 다 바꿔봐도
    수렴이 느려서 다음으로 시도해보는 파라미터).

    mpc_rate_hz는 optimization_dt(=interpolation_steps/mpc_rate_hz)를 통해
    MPC가 내부적으로 가정하는 "tick 간 경과 시간"을 정한다. 코드가 원래
    MIT_RATE_HZ(120Hz, 8.3ms)를 가정했는데, 실측(RTX 3050 Ti)으로는 plan()이
    평균 14ms 걸려 루프가 실제로 ~60Hz로만 돈다 — 이 불일치 때문에 MPC의
    속도/가속도 적분(운동학 가정)이 실제 경과 시간과 안 맞아 궤적 품질이
    나빠질 수 있다. 실측 rate(예: 60.0)로 맞추면 이 불일치가 없어진다.
    """
    import torch
    from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
    from curobo.types import DeviceCfg, GoalToolPose, JointState
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
    from curobo._src.types.robot import RobotCfg

    optimizer_configs: list = ["mpc/lbfgs_mpc.yml"]
    if accel_regularization_scale != 1.0:
        import os

        from curobo._src.util.config_io import load_yaml
        from curobo.content import get_task_configs_path

        # plain yaml.safe_load()는 curobo가 자체 패치한 float 파서(예: "1e-3"처럼
        # 소수점 없는 지수 표기)를 못 알아듣고 문자열로 남겨버려, 나중에 CUDA
        # 커널 인자로 들어갈 때 TypeError로 죽는다 — 반드시 curobo의 load_yaml
        # (patched Loader)을 써야 한다.
        base_yaml = os.path.join(str(get_task_configs_path()), "mpc", "lbfgs_mpc.yml")
        mpc_cost_cfg = load_yaml(base_yaml)
        # squared_l2_regularization_weight = [velocity, acceleration, jerk, torque_smooth, energy_smooth]
        cspace_reg = mpc_cost_cfg["rollout"]["cost_cfg"]["cspace_cfg"]["squared_l2_regularization_weight"]
        cspace_reg[1] = cspace_reg[1] * accel_regularization_scale
        print(f"  [cuRobo v2 MPC] 가속도 regularization weight: "
              f"{cspace_reg[1] / accel_regularization_scale:.1f} → {cspace_reg[1]:.1f} "
              f"(x{accel_regularization_scale})")
        optimizer_configs = [mpc_cost_cfg]

    device_cfg = DeviceCfg()
    kin_cfg = KinematicsCfg.from_robot_yaml_file(
        xrdf_path, tool_frames=["flange_link"],
        device_cfg=device_cfg, urdf_path=urdf_path,
    )
    robot_cfg = RobotCfg(kinematics=kin_cfg, device_cfg=device_cfg)
    # "table" cuboid는 Z=0 근처가 바닥(책상면)인 horizontal 설치를 가정하고
    # 넣은 가상 충돌체다. right/left처럼 옆으로 눕혀 설치되면 base Z축이
    # 더 이상 수직이 아니므로(실측: right에서 Z축은 "오른쪽"), 이 2m x 2m
    # 판이 실제로는 바닥이 아니라 작업공간을 가로지르는 가상의 벽이 되어
    # 특정 방향 이동을 collision cost(가중치 10000)로 계속 방해할 수 있다.
    cuboids = {"table": {"dims": [2.0, 2.0, 0.05], "pose": [0.0, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0]}}
    interpolation_steps = 4  # cuRobo 고정값; command_dt = optimization_dt / interpolation_steps
    mpc_cfg = ModelPredictiveControlCfg.create(
        robot=robot_cfg,
        scene_model={"cuboid": cuboids} if enable_scene_collision else None,
        collision_cache={"cuboid": 1} if enable_scene_collision else None,
        device_cfg=device_cfg,
        self_collision_check=enable_self_collision,
        use_cuda_graph=use_cuda_graph,
        optimization_dt=interpolation_steps / mpc_rate_hz,
        interpolation_steps=interpolation_steps,
        use_deceleration_on_failure=use_deceleration_on_failure,
        optimizer_configs=optimizer_configs,
    )
    mpc = ModelPredictiveControl(mpc_cfg)
    print("[cuRobo v2 MPC] warming up...")

    tool_frame = mpc.tool_frames[0]
    setup_done = False
    last_goal_eef: list[float] | None = None
    last_velocity: "torch.Tensor | None" = None
    last_acceleration: "torch.Tensor | None" = None
    debug_tick = 0

    def plan(q_current: list[float], target_eef: list[float], reset: bool = False):
        nonlocal setup_done, last_goal_eef, last_velocity, last_acceleration, debug_tick
        q_t = torch.tensor([q_current], dtype=device_cfg.dtype, device=device_cfg.device)
        current_state = JointState.from_position(q_t, joint_names=mpc.joint_names)
        # MPC's warm-started B-spline transition model expects continuity
        # with the velocity/acceleration it commanded last tick. Re-zeroing
        # both every call (JointState.from_position's default) tells the
        # optimizer "you're at rest" even mid-trajectory, which stalls
        # convergence almost completely (verified on GPU: with this fix a
        # nearby goal converges in ~2s; without it, error stays put).
        reset_velocity = (not setup_done) or reset
        if not reset_velocity and last_velocity is not None:
            current_state.velocity = last_velocity
            current_state.acceleration = last_acceleration

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
        debug_tick += 1
        if debug and debug_tick % 30 == 0:
            pos_err = result.position_error
            rot_err = result.rotation_error
            pos_err_v = float(pos_err.flatten()[0]) if pos_err is not None else float("nan")
            rot_err_v = float(rot_err.flatten()[0]) if rot_err is not None else float("nan")
            v_norm = (
                float(last_velocity.abs().max()) if last_velocity is not None else float("nan")
            )
            print(
                f"    [mpc-debug] tick={debug_tick}  success={bool(result.success.any()) if result.success is not None else None}  "
                f"feasible={result.feasible}  optimizer position_error={pos_err_v*1000:.1f}mm  "
                f"rotation_error={rot_err_v:.4f}  max|v_des|={v_norm:.4f} rad/s"
            )
        if result.action_sequence is None:
            last_velocity = None
            last_acceleration = None
            return None
        last_velocity = result.action_sequence.velocity[:, -1, :].detach()
        last_acceleration = result.action_sequence.acceleration[:, -1, :].detach()
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


def _make_planner(
    urdf_path: str,
    use_deceleration_on_failure: bool = True,
    use_cuda_graph: bool = True,
    accel_regularization_scale: float = 1.0,
    mpc_rate_hz: float = MIT_RATE_HZ,
    enable_scene_collision: bool = True,
    enable_self_collision: bool = True,
    debug: bool = False,
):
    if _module_available("curobo.geom.sdf.world"):
        return _make_v1_planner(urdf_path)
    if _module_available("curobo.model_predictive_control"):
        xrdf = urdf_path.replace(".urdf", ".xrdf")
        return _make_v2_mpc_planner(
            urdf_path, xrdf,
            use_deceleration_on_failure=use_deceleration_on_failure,
            use_cuda_graph=use_cuda_graph,
            accel_regularization_scale=accel_regularization_scale,
            mpc_rate_hz=mpc_rate_hz,
            enable_scene_collision=enable_scene_collision,
            enable_self_collision=enable_self_collision,
            debug=debug,
        )
    raise RuntimeError(
        "cuRobo v1 (MotionGen, curobo.geom.sdf.world) 또는 v2 MPC "
        "(curobo.model_predictive_control)가 필요합니다."
    )


def _send_mit(
    api: PiperAPI,
    torques: list[float],
    p_des: list[float],
    v_des: list[float],
    kp_list: list[float],
    kd: float,
) -> None:
    """6-joint MIT command with per-joint kp (J1-J3 body / J4-J6 wrist)."""
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
    """kp를 0.5 → kp_target(J1-J3)/kp_wrist(J4-J6)으로 geomspace 램프 후 hold_pos 유지.

    MIT 진입 직후 목표 kp로 바로 뛰면 순간적인 토크 스냅/진동이 생길 수 있어
    서서히 올린다 (test_mit_controller2.py의 _ramp_gains와 동일한 방식).
    """
    if ramp_time <= 0.0:
        return

    ramp_steps = max(2, int(round(ramp_time * MIT_RATE_HZ)))
    body_start = min(0.5, kp_target)
    wrist_start = min(0.5, kp_wrist)
    body_sched = np.geomspace(body_start, kp_target, ramp_steps)
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


# ── MIT 루프로 trajectory 실행 ───────────────────────────────────────────────
def execute_trajectory(
    api: PiperAPI,
    grav_comp: MujocoGravityCompensator,
    positions,        # (N, 6) ndarray
    velocities,       # (N, 6) ndarray
    kp: float,
    kp_wrist: float,
    kd: float,
    torque_scale: float,
    timeout_s: float,
    joint_torque_scales: list[float] | None = None,
) -> bool:
    """trajectory를 MIT 120 Hz 루프로 실행. 완료 반환 True, 타임아웃 False."""
    kp_list = [kp, kp, kp, kp_wrist, kp_wrist, kp_wrist]
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

        grav = grav_comp.compute(
            q, joint_velocities=v,
            torque_scale=torque_scale,
            joint_torque_scales=joint_torque_scales,
        )

        # ── 실제 q에 가장 가까운 waypoint에서 시작 (최초 1회) ──────────────
        # (plan은 q_pre에서 시작하므로 실행 시점 q와 다를 수 있음)
        # idx==0일 때 한 번 seek 한다.
        if idx == 0:
            dists = np.linalg.norm(positions - np.array(q), axis=1)
            idx = int(np.argmin(dists))
            # 역방향 seek 방지: 앞의 waypoint만 허용
            idx = min(idx, n_waypoints // 4)

        p_des = positions[idx].tolist()
        v_des = velocities[idx].tolist()

        _send_mit(api, grav.command_torques, p_des, v_des, kp_list, kd)
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
    kp_wrist: float,
    kd: float,
    torque_scale: float,
    timeout_s: float,
    joint_torque_scales: list[float] | None = None,
    pos_tol_m: float = 0.005,
    settle_s: float = 0.3,
) -> bool:
    """MPC를 매 tick 재호출해 목표에 도달할 때까지 실행 (closed-loop).

    MotionGen과 달리 MPC는 한 번의 plan()이 전체 궤적을 주지 않으므로
    (receding-horizon), 매 120Hz tick마다 현재 q로 다시 plan()을 호출하고
    그 결과의 마지막 waypoint를 MIT desired position/velocity로 사용한다.
    EEF 위치 오차가 pos_tol_m 이내로 settle_s동안 유지되면 성공.
    """
    kp_list = [kp, kp, kp, kp_wrist, kp_wrist, kp_wrist]
    period_ns = int(round(1_000_000_000 / MIT_RATE_HZ))
    deadline = time.monotonic() + timeout_s
    next_ns = time.monotonic_ns()
    converged_since: float | None = None

    t_loop_start = time.monotonic()
    n_ticks = 0
    plan_time_total = 0.0
    plan_time_max = 0.0

    def _report_tick_rate() -> None:
        elapsed = time.monotonic() - t_loop_start
        if n_ticks == 0 or elapsed <= 0:
            return
        achieved_hz = n_ticks / elapsed
        mean_plan_ms = (plan_time_total / n_ticks) * 1000
        print(
            f"    [tick] {n_ticks} ticks in {elapsed:.2f}s → 실측 {achieved_hz:.1f} Hz "
            f"(목표 {MIT_RATE_HZ:.0f} Hz)  plan() 평균 {mean_plan_ms:.1f}ms  "
            f"최대 {plan_time_max*1000:.1f}ms"
        )

    while True:
        now = time.monotonic()
        if now > deadline:
            print("  [!] timeout")
            _report_tick_rate()
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
        grav = grav_comp.compute(
            q, joint_velocities=v,
            torque_scale=torque_scale,
            joint_torque_scales=joint_torque_scales,
        )

        t_plan0 = time.monotonic()
        planned = plan(q, target_eef)
        plan_dt = time.monotonic() - t_plan0
        n_ticks += 1
        plan_time_total += plan_dt
        plan_time_max = max(plan_time_max, plan_dt)

        if planned is None:
            p_des, v_des = q, [0.0] * 6
        else:
            pos, vel = planned
            p_des, v_des = pos[-1].tolist(), vel[-1].tolist()

        _send_mit(api, grav.command_torques, p_des, v_des, kp_list, kd)

        eef_now = api.get_eef_pose()
        if eef_now is not None:
            pos_err = math.sqrt(sum((eef_now[i] - target_eef[i]) ** 2 for i in range(3)))
            if pos_err < pos_tol_m:
                if converged_since is None:
                    converged_since = now
                elif now - converged_since >= settle_s:
                    _report_tick_rate()
                    return True
            else:
                converged_since = None

        next_ns += period_ns
        while next_ns < time.monotonic_ns():
            next_ns += period_ns


def _angle_wrap(diff_rad: float) -> float:
    """각도 차이를 (-pi, pi] 범위로 정규화 (±180° 경계 wraparound 방지)."""
    return (diff_rad + math.pi) % (2 * math.pi) - math.pi


def execute_mpc_reach_continuous(
    api: PiperAPI,
    grav_comp: MujocoGravityCompensator,
    plan,
    target_eef: list[float],
    kp: float,
    kp_wrist: float,
    kd: float,
    torque_scale: float,
    timeout_s: float,
    joint_torque_scales: list[float] | None = None,
    carrot_speed_m_s: float = 0.05,
    carrot_speed_rad_s: float = math.radians(15.0),
    pos_tol_m: float = 0.005,
    settle_s: float = 0.3,
    goal_update_threshold_m: float = 0.005,
) -> bool:
    """target_eef까지 '움직이는 당근'(moving carrot) 방식으로 MPC를 실행.

    execute_mpc_reach_via_subgoals()는 정적인 sub-goal을 주고 최대
    subgoal_timeout_s초까지 수렴을 기다리는데, 이 MPC(horizon=1, teleop용
    튜닝)는 "가만히 있는 먼 목표"에는 잘 반응하지 않고 아주 작은 크루즈
    속도로만 창(--mpc-debug로 실측: 0.03~0.05 rad/s에서 고정)에 수렴한다.

    실제 teleop(사람 손 추적)은 목표 자체가 매 tick(25Hz) 계속 갱신되며
    가까운 거리만 유지되는데, 이 함수는 그 패턴을 그대로 재현한다.

    plan()에 실제로 넘기는 목표(committed goal)는 carrot을 그대로 매 tick
    넘기지 않고, carrot이 goal_update_threshold_m 이상 움직였을 때만
    갱신한다. plan() 내부의 "target_eef != last_goal_eef일 때만
    update_goal_tool_poses() 재호출" 로직은 같은 목표를 여러 tick 유지해야
    (단일 시드 IK가) 점진적으로 개선된다는 전제인데, carrot을 매 tick(0.1mm
    단위로) 그대로 넘기면 그 전제가 깨져 매 tick 새 IK 시드로 다시 시작하게
    된다 — 실측으로 팔이 완전히 멈춰버리는 현상과 연관된 것으로 의심됨.
    """
    kp_list = [kp, kp, kp, kp_wrist, kp_wrist, kp_wrist]
    period_ns = int(round(1_000_000_000 / MIT_RATE_HZ))
    dt = 1.0 / MIT_RATE_HZ
    deadline = time.monotonic() + timeout_s
    next_ns = time.monotonic_ns()
    converged_since: float | None = None

    eef_now = api.get_eef_pose()
    if eef_now is None:
        print("  [!] 현재 EEF pose 읽기 실패.")
        return False
    carrot = list(eef_now)
    committed_goal = list(carrot)

    n_ticks = 0
    n_goal_updates = 0
    t_loop_start = time.monotonic()
    last_report = t_loop_start

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
        grav = grav_comp.compute(
            q, joint_velocities=v,
            torque_scale=torque_scale,
            joint_torque_scales=joint_torque_scales,
        )

        # carrot을 target_eef 쪽으로 이번 tick만큼 전진 (사람 손이 목표로
        # 꾸준히 이동하는 것을 흉내냄).
        pos_diff = [target_eef[i] - carrot[i] for i in range(3)]
        pos_dist = math.sqrt(sum(d * d for d in pos_diff))
        step = min(pos_dist, carrot_speed_m_s * dt)
        if pos_dist > 1e-9:
            for i in range(3):
                carrot[i] += pos_diff[i] / pos_dist * step
        rot_diff = [_angle_wrap(target_eef[3 + i] - carrot[3 + i]) for i in range(3)]
        rot_dist = math.sqrt(sum(d * d for d in rot_diff))
        rstep = min(rot_dist, carrot_speed_rad_s * dt)
        if rot_dist > 1e-9:
            for i in range(3):
                carrot[3 + i] += rot_diff[i] / rot_dist * rstep

        # carrot이 충분히(threshold 이상) 전진했을 때만 실제 plan() 목표를
        # 갱신한다 — 같은 목표를 여러 tick 유지해야 IK 시드가 정착한다.
        committed_dist = math.sqrt(
            sum((carrot[i] - committed_goal[i]) ** 2 for i in range(3))
        )
        if committed_dist >= goal_update_threshold_m:
            committed_goal = list(carrot)
            n_goal_updates += 1

        planned = plan(q, committed_goal)
        n_ticks += 1
        if planned is None:
            p_des, v_des = q, [0.0] * 6
        else:
            pos, vel = planned
            p_des, v_des = pos[-1].tolist(), vel[-1].tolist()

        _send_mit(api, grav.command_torques, p_des, v_des, kp_list, kd)

        eef_actual = api.get_eef_pose()
        if eef_actual is not None:
            pos_err_target = math.sqrt(
                sum((eef_actual[i] - target_eef[i]) ** 2 for i in range(3))
            )
            if now - last_report >= 1.0:
                pos_err_carrot = math.sqrt(
                    sum((eef_actual[i] - carrot[i]) ** 2 for i in range(3))
                )
                v_measured = max(abs(x) for x in v)
                v_des_max = max(abs(x) for x in v_des)
                print(
                    f"    [carrot] t={now - t_loop_start:5.1f}s  "
                    f"eef=[{eef_actual[0]*100:+.1f},{eef_actual[1]*100:+.1f},{eef_actual[2]*100:+.1f}]cm  "
                    f"carrot=[{carrot[0]*100:+.1f},{carrot[1]*100:+.1f},{carrot[2]*100:+.1f}]cm  "
                    f"|eef-carrot|={pos_err_carrot*1000:.1f}mm  |eef-target|={pos_err_target*1000:.1f}mm  "
                    f"goal_updates={n_goal_updates}  max|측정v|={v_measured:.4f}rad/s  max|v_des|={v_des_max:.4f}rad/s"
                )
                last_report = now
            if pos_err_target < pos_tol_m:
                if converged_since is None:
                    converged_since = now
                elif now - converged_since >= settle_s:
                    return True
            else:
                converged_since = None

        next_ns += period_ns
        while next_ns < time.monotonic_ns():
            next_ns += period_ns


def _interpolate_eef_waypoints(
    start_eef: list[float],
    end_eef: list[float],
    max_step_m: float,
    max_step_rad: float,
) -> list[list[float]]:
    """start_eef→end_eef를 max_step_m/max_step_rad 이내 간격으로 선형 보간.

    cuRobo MPC는 horizon=1에 non-terminal 가중치가 아주 작아 매 tick 가까운
    목표를 계속 갱신받는 teleop 용도로 튜닝돼 있다. 수십 cm 떨어진 목표를
    한 번에 goal로 넣으면 사실상 수렴하지 않으므로, teleop처럼 작은 조각
    (roll/pitch/yaw는 단순 선형보간 — 몇 도 단위라 gimbal 문제 없음)으로
    나눠 순차적으로 먹인다. 반환값에는 시작점은 빠지고 끝점까지 포함된다.
    """
    pos_diff = [end_eef[i] - start_eef[i] for i in range(3)]
    pos_dist = math.sqrt(sum(d * d for d in pos_diff))

    rot_diff = [_angle_wrap(end_eef[i] - start_eef[i]) for i in range(3, 6)]
    rot_dist = math.sqrt(sum(d * d for d in rot_diff))

    n_steps = max(
        1,
        math.ceil(pos_dist / max_step_m) if max_step_m > 0 else 1,
        math.ceil(rot_dist / max_step_rad) if max_step_rad > 0 else 1,
    )

    waypoints = []
    for step in range(1, n_steps + 1):
        frac = step / n_steps
        wp = [start_eef[i] + pos_diff[i] * frac for i in range(3)]
        wp += [start_eef[3 + i] + rot_diff[i] * frac for i in range(3)]
        waypoints.append(wp)
    return waypoints


def execute_mpc_reach_via_subgoals(
    api: PiperAPI,
    grav_comp: MujocoGravityCompensator,
    plan,
    target_eef: list[float],
    kp: float,
    kp_wrist: float,
    kd: float,
    torque_scale: float,
    timeout_s: float,
    joint_torque_scales: list[float] | None = None,
    subgoal_step_m: float = 0.03,
    subgoal_step_rad: float = math.radians(5.0),
    subgoal_timeout_s: float = 3.0,
    pos_tol_m: float = 0.005,
    settle_s: float = 0.3,
) -> bool:
    """target_eef까지 작은 sub-goal 조각으로 나눠 순차적으로 MPC 추적.

    execute_mpc_reach()를 target_eef 하나로 직접 부르면 MPC가 먼 목표에는
    사실상 수렴하지 않는다 (하드웨어/소프트웨어 양쪽에서 확인됨). teleop이
    매 tick 가까운 목표를 넘겨주는 것과 같은 방식으로, 현재 EEF에서 target까지
    subgoal_step_m/subgoal_step_rad 이내 간격의 중간 목표들을 만들어 순서대로
    execute_mpc_reach()에 넘긴다. 마지막 sub-goal만 pos_tol_m/settle_s로
    엄격하게 정착을 확인하고, 중간 sub-goal은 다음 조각으로 빨리 넘어가도록
    느슨한 허용오차와 짧은 settle을 쓴다. 전체 timeout_s는 모든 sub-goal에
    걸친 합산 제한이다.
    """
    eef_now = api.get_eef_pose()
    if eef_now is None:
        print("  [!] 현재 EEF pose 읽기 실패.")
        return False

    subgoals = _interpolate_eef_waypoints(eef_now, target_eef, subgoal_step_m, subgoal_step_rad)
    print(f"  [subgoal] {len(subgoals)}개 중간 목표로 분할 "
          f"(step<={subgoal_step_m*100:.1f}cm/{math.degrees(subgoal_step_rad):.1f}°)")

    deadline = time.monotonic() + timeout_s
    for i, sub_eef in enumerate(subgoals):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"  [!] 전체 timeout (sub-goal {i+1}/{len(subgoals)} 진행 전)")
            return False

        is_final = i == len(subgoals) - 1
        step_timeout = remaining if is_final else min(subgoal_timeout_s, remaining)
        ok = execute_mpc_reach(
            api, grav_comp, plan, sub_eef,
            kp=kp, kp_wrist=kp_wrist, kd=kd,
            torque_scale=torque_scale,
            timeout_s=step_timeout,
            joint_torque_scales=joint_torque_scales,
            pos_tol_m=pos_tol_m if is_final else max(pos_tol_m, subgoal_step_m * 0.75),
            settle_s=settle_s if is_final else 0.05,
        )
        if not ok:
            if is_final:
                print(f"  [!] sub-goal {i+1}/{len(subgoals)} 도달 실패 (timeout)")
                return False
            # 중간 sub-goal은 느슨한 허용오차에도 못 미쳤더라도 전체를
            # 중단하지 않고 다음(목표에 더 가까운) sub-goal로 계속 진행한다.
            # MPC는 receding-horizon이라 목표가 계속 갱신되는 한 이전
            # sub-goal에서 못 채운 진행분을 다음 구간에서 이어서 좁혀간다
            # (관측됨: 이전엔 첫 sub-goal 실패로 전체를 포기해 0% 진행에
            # 그쳤던 이동이, 계속 진행할 경우 뒤에서 상당 부분 만회함).
            print(f"  [!] sub-goal {i+1}/{len(subgoals)} 허용오차 못 미침 (timeout) — 다음 sub-goal로 계속 진행")

    return True


def hold_position(
    api: PiperAPI,
    grav_comp: MujocoGravityCompensator,
    hold_s: float,
    kp: float,
    kp_wrist: float,
    kd: float,
    torque_scale: float,
    joint_torque_scales: list[float] | None = None,
) -> list[float] | None:
    """hold_s 동안 현재 위치 유지. 마지막 q 반환."""
    kp_list = [kp, kp, kp, kp_wrist, kp_wrist, kp_wrist]
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

        grav = grav_comp.compute(
            q, joint_velocities=v,
            torque_scale=torque_scale,
            joint_torque_scales=joint_torque_scales,
        )
        _send_mit(api, grav.command_torques, hold_pos, [0.0] * 6, kp_list, kd)
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

    # ── cuRobo 초기화 ────────────────────────────────────────────────────────
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

    # ── cuRobo v2 MPC 사전 워밍업 ────────────────────────────────────────────
    # mpc.setup()(CUDA graph capture 등 무거운 1회성 초기화)은 plan()의 첫
    # 실제 호출 시점에 지연 실행된다. 여기서 미리 트리거해두지 않으면 그
    # 비용이 첫 waypoint의 timeout 예산을 통째로 먹어 첫 이동이 0% 진행률로
    # 타임아웃난다 (test_mit_controller2.py에서 실측됨). 목표를 현재 위치로
    # 줘서 실제 이동은 발생시키지 않는다.
    if is_continuous:
        js_warm = api.get_joint_state()
        eef_warm = api.get_eef_pose()
        if js_warm is not None and eef_warm is not None:
            print("[cuRobo v2 MPC] 사전 워밍업 중...", end="", flush=True)
            t_warm = time.monotonic()
            plan(list(js_warm.joints), eef_warm, reset=True)
            print(f" done ({time.monotonic()-t_warm:.1f}s)")

    # ── MIT 모드 진입 ────────────────────────────────────────────────────────
    api.enable()
    api.set_mit_mode()
    print("\nMIT mode enabled.\n")
    js_mit0 = api.get_joint_state()
    if js_mit0 is not None:
        _ramp_gains(
            api, grav_comp, list(js_mit0.joints),
            args.kp, kp_wrist, args.kd, args.torque_scale, args.ramp_time,
            joint_torque_scales,
        )

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
            # MPC: 목표가 멀면 사실상 수렴하지 않으므로 (검증됨) teleop처럼
            # 작은 sub-goal 조각으로 나눠 순차적으로 먹인다.
            print("  [cuRobo MPC] 실행 중 (sub-goal closed-loop)...")
            t_exec = time.monotonic()
            ok = execute_mpc_reach_via_subgoals(
                api, grav_comp, plan, target_eef,
                kp=args.kp, kp_wrist=kp_wrist, kd=args.kd,
                torque_scale=args.torque_scale,
                timeout_s=args.timeout,
                joint_torque_scales=joint_torque_scales,
                subgoal_step_m=args.subgoal_step_m,
                subgoal_step_rad=math.radians(args.subgoal_step_deg),
                subgoal_timeout_s=args.subgoal_timeout,
            )
            t_exec = time.monotonic() - t_exec
            print(f"  {'완료' if ok else 'TIMEOUT'}  ({t_exec*1000:.0f}ms)")
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
            diff0 = np.array(q_current) - pos[0]
            print(f"  q_current vs pos[0] diff: {[f'{d:+.4f}' for d in diff0.tolist()]}")
            if np.linalg.norm(diff0) > 0.3:
                print("  [경고] traj 시작점과 현재 위치가 0.3 rad 이상 떨어져 있음!")

            # MIT 실행
            print("  실행 중...", end="", flush=True)
            t_exec = time.monotonic()
            ok = execute_trajectory(
                api, grav_comp, pos, vel,
                kp=args.kp, kp_wrist=kp_wrist, kd=args.kd,
                torque_scale=args.torque_scale,
                timeout_s=args.timeout,
                joint_torque_scales=joint_torque_scales,
            )
            t_exec = time.monotonic() - t_exec
            print(f" {'완료' if ok else 'TIMEOUT'}  ({t_exec*1000:.0f}ms)")

        # 홀드 + 오차 측정
        print(f"  {HOLD_AFTER_REACH_S:.1f}s 홀드 중...", end="", flush=True)
        q_final = hold_position(
            api, grav_comp, HOLD_AFTER_REACH_S,
            args.kp, kp_wrist, args.kd, args.torque_scale,
            joint_torque_scales=joint_torque_scales,
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
    hold_position(
        api, grav_comp, 1.0, args.kp, kp_wrist, args.kd, args.torque_scale,
        joint_torque_scales=joint_torque_scales,
    )
    api.disconnect()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCtrl+C — 종료.")
        raise SystemExit(0)
