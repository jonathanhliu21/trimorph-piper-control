"""cuRobo MotionGen worker: 25 Hz EEF pose → 120 Hz joint trajectory."""
from __future__ import annotations

import math
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

    table_cuboid = {
        "name": "table",
        "dims": [2.0, 2.0, 0.05],
        "pose": [0.0, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0],
    }
    all_cuboids = [table_cuboid] + list(config.curobo_world_cuboids)
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
    print("[curobo_worker] Warming up MotionGen (JIT compile)…")
    motion_gen.warmup()
    print("[curobo_worker] Warmup complete.")

    plan_config = MotionGenPlanConfig(
        max_attempts=4,
        enable_graph=True,
        enable_opt=True,
    )

    init_done.set()

    input_period_s = 1.0 / config.curobo_input_hz
    last_eef: list[float] | None = None

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
        x, y, z, roll, pitch, yaw = target_eef
        w_q, x_q, y_q, z_q = _euler_xyz_to_quat_wxyz(roll, pitch, yaw)

        position = torch.tensor([[x, y, z]], dtype=torch.float32, device=tensor_args.device)
        # cuRobo quaternion convention: (w, x, y, z)
        quaternion = torch.tensor([[w_q, x_q, y_q, z_q]], dtype=torch.float32, device=tensor_args.device)
        goal_pose = Pose(position=position, quaternion=quaternion)

        q_tensor = torch.tensor([q_current], dtype=torch.float32, device=tensor_args.device)
        start_state = JointState.from_position(q_tensor)

        try:
            result = motion_gen.plan_single(start_state, goal_pose, plan_config)
            if result.success.item():
                pos = result.interpolated_plan.position.cpu().numpy()   # (N, 6)
                vel = result.interpolated_plan.velocity.cpu().numpy()   # (N, 6)
                trajectory_buffer.push_trajectory(pos, vel, dt_s=1.0 / output_hz)
            # On failure, buffer keeps current trajectory until exhausted → hold fallback
        except Exception as exc:
            print(f"[curobo_worker] plan_single error: {exc}")

        elapsed = time.monotonic() - loop_start
        shared.shutdown.wait(max(0.0, input_period_s - elapsed))
