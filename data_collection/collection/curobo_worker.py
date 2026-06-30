"""cuRobo MotionGen worker: 25 Hz EEF pose → 120 Hz joint trajectory."""
from __future__ import annotations

import math
import importlib.util
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from scripts.piper_api.curobo_trajectory_buffer import CuRoboTrajectoryBuffer

from .config import CollectionConfig
from .episode import RecordingManager
from .shared_state import SharedState

if TYPE_CHECKING:
    pass


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> list[float]:
    """Convert extrinsic XYZ Euler angles to quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def _world_cuboids(config: CollectionConfig) -> list[dict]:
    table_cuboid = {
        "name": "table",
        "dims": [2.0, 2.0, 0.05],
        "pose": [0.0, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0],
    }
    return [table_cuboid] + list(config.curobo_world_cuboids)


def _v2_scene_model(cuboids: list[dict]) -> dict:
    scene_cuboids = {}
    for index, cuboid in enumerate(cuboids):
        name = str(cuboid.get("name") or f"cuboid_{index}")
        scene_cuboids[name] = {
            "dims": list(cuboid["dims"]),
            "pose": list(cuboid["pose"]),
        }
    return {"cuboid": scene_cuboids}


def _joint_state_to_arrays(joint_state) -> tuple[np.ndarray, np.ndarray]:
    position = joint_state.position.detach()
    velocity = (
        joint_state.velocity.detach()
        if joint_state.velocity is not None
        else position.new_zeros(position.shape)
    )
    while position.ndim > 2:
        position = position[0]
        velocity = velocity[0]
    if position.ndim == 1:
        position = position.unsqueeze(0)
        velocity = velocity.unsqueeze(0)
    return position.cpu().numpy(), velocity.cpu().numpy()


def _create_v1_planner(config: CollectionConfig):
    import torch
    from curobo.geom.sdf.world import WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState, RobotConfig
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
    )

    tensor_args = TensorDeviceType()

    robot_cfg = RobotConfig.from_basic(
        urdf_path=str(config.curobo_urdf_path),
        base_link="base_link",
        ee_link="flange_link",
        tensor_args=tensor_args,
    )

    all_cuboids = _world_cuboids(config)
    world_cfg = WorldConfig.from_dict({"cuboid": all_cuboids})

    output_hz = config.curobo_output_hz
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        interpolation_dt=1.0 / output_hz,
        use_cuda_graph=True,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
    )
    motion_gen = MotionGen(motion_gen_cfg)
    print("[curobo_worker] Using cuRobo v1 MotionGen.")
    print("[curobo_worker] Warming up MotionGen (JIT compile)...")
    motion_gen.warmup()
    print("[curobo_worker] Warmup complete.")

    plan_config = MotionGenPlanConfig(
        max_attempts=4,
        enable_graph=True,
        enable_opt=True,
    )

    def plan(q_current: list[float], target_eef: list[float]):
        x, y, z, roll, pitch, yaw = target_eef
        w_q, x_q, y_q, z_q = _euler_xyz_to_quat_wxyz(roll, pitch, yaw)
        position = torch.tensor(
            [[x, y, z]], dtype=torch.float32, device=tensor_args.device
        )
        quaternion = torch.tensor(
            [[w_q, x_q, y_q, z_q]], dtype=torch.float32, device=tensor_args.device
        )
        goal_pose = Pose(position=position, quaternion=quaternion)
        q_tensor = torch.tensor([q_current], dtype=torch.float32, device=tensor_args.device)
        start_state = JointState.from_position(q_tensor)
        result = motion_gen.plan_single(start_state, goal_pose, plan_config)
        if not result.success.item():
            return None
        return (
            result.interpolated_plan.position.cpu().numpy(),
            result.interpolated_plan.velocity.cpu().numpy(),
        )

    return plan


def _create_v2_planner(config: CollectionConfig):
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import DeviceCfg, GoalToolPose, JointState
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
    from curobo._src.types.robot import RobotCfg

    device_cfg = DeviceCfg()
    kinematics_cfg = KinematicsCfg.from_robot_yaml_file(
        str(config.curobo_xrdf_path),
        tool_frames=["flange_link"],
        device_cfg=device_cfg,
        urdf_path=str(config.curobo_urdf_path),
    )
    robot_cfg = RobotCfg(kinematics=kinematics_cfg, device_cfg=device_cfg)
    all_cuboids = _world_cuboids(config)
    planner_cfg = MotionPlannerCfg.create(
        robot=robot_cfg,
        scene_model=_v2_scene_model(all_cuboids),
        collision_cache={"cuboid": max(1, len(all_cuboids))},
        device_cfg=device_cfg,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        self_collision_check=True,
        use_cuda_graph=True,
    )
    planner = MotionPlanner(planner_cfg)
    print("[curobo_worker] Using cuRobo v2 MotionPlanner.")
    print("[curobo_worker] Warming up MotionPlanner (JIT compile)...")
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    print("[curobo_worker] Warmup complete.")

    def plan(q_current: list[float], target_eef: list[float]):
        x, y, z, roll, pitch, yaw = target_eef
        w_q, x_q, y_q, z_q = _euler_xyz_to_quat_wxyz(roll, pitch, yaw)
        current_state = JointState.from_position(
            torch.tensor(
                [q_current],
                dtype=device_cfg.dtype,
                device=device_cfg.device,
            ),
            joint_names=planner.joint_names,
        )
        goal = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=torch.tensor(
                [[[[[x, y, z]]]]],
                dtype=device_cfg.dtype,
                device=device_cfg.device,
            ),
            quaternion=torch.tensor(
                [[[[[w_q, x_q, y_q, z_q]]]]],
                dtype=device_cfg.dtype,
                device=device_cfg.device,
            ),
        )
        result = planner.plan_pose(goal, current_state, max_attempts=4)
        if result is None or not result.success.any().item():
            return None
        return _joint_state_to_arrays(result.get_interpolated_plan())

    return plan


def _create_curobo_planner(config: CollectionConfig):
    if _module_available("curobo.geom.sdf.world"):
        return _create_v1_planner(config)
    if _module_available("curobo.motion_planner"):
        return _create_v2_planner(config)
    raise RuntimeError(
        "Installed curobo package exposes neither v1 MotionGen nor v2 MotionPlanner."
    )


def run_curobo_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    trajectory_buffer: CuRoboTrajectoryBuffer,
    *,
    init_done: threading.Event,
) -> None:
    """
    GPU thread: polls SharedState for a new EEF target at CUROBO_INPUT_HZ,
    runs cuRobo MotionGen, and pushes the interpolated joint trajectory
    (sampled at CUROBO_OUTPUT_HZ) into trajectory_buffer.
    """
    plan = _create_curobo_planner(config)

    init_done.set()

    input_period_s = 1.0 / config.curobo_input_hz
    last_eef: list[float] | None = None
    _diag_plan_count = 0

    while not shared.shutdown.is_set():
        loop_start = time.monotonic()

        if not shared.teleop_enabled():
            trajectory_buffer.reset()
            elapsed = time.monotonic() - loop_start
            shared.shutdown.wait(max(0.0, input_period_s - elapsed))
            continue

        target_eef = shared.latest_target_eef()
        if target_eef is None or target_eef == last_eef:
            elapsed = time.monotonic() - loop_start
            shared.shutdown.wait(max(0.0, input_period_s - elapsed))
            continue

        q_current = shared.latest_joint_positions()
        if q_current is None:
            elapsed = time.monotonic() - loop_start
            shared.shutdown.wait(max(0.0, input_period_s - elapsed))
            continue

        last_eef = target_eef

        try:
            planned = plan(q_current, target_eef)
            if planned is not None:
                pos, vel = planned
                _diag_plan_count += 1
                if _diag_plan_count <= 5:
                    print(
                        f"[diag/cuRobo #{_diag_plan_count}] Plan OK  "
                        f"tgt_eef={[f'{v:+.3f}' for v in target_eef]}  "
                        f"waypoints={len(pos)}  "
                        f"q_start={[f'{v:+.3f}' for v in pos[0].tolist()]}  "
                        f"q_end={[f'{v:+.3f}' for v in pos[-1].tolist()]}"
                    )
                trajectory_buffer.push_trajectory(pos, vel, dt_s=1.0 / config.curobo_output_hz)
            else:
                print(
                    f"[diag/cuRobo] Plan FAILED  tgt_eef={[f'{v:+.3f}' for v in target_eef]}  "
                    f"q_current={[f'{v:+.3f}' for v in q_current]}"
                )
        except Exception as exc:
            print(f"[curobo_worker] plan_single error: {exc}")

        elapsed = time.monotonic() - loop_start
        shared.shutdown.wait(max(0.0, input_period_s - elapsed))
