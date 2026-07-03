"""cuRobo worker: 25 Hz EEF pose -> 120 Hz joint trajectory.

Backend is chosen by what the installed curobo package exposes:
  - v1 MotionGen (curobo.geom.sdf.world): plans a fresh trajectory only when
    the EEF target changes, then the interpolated plan is drained by the
    120 Hz MIT loop.
  - v2 ModelPredictiveControl (curobo.model_predictive_control): re-optimizes
    every tick using live joint feedback (closed-loop), and only the target
    goal pose is updated when the EEF target changes.
"""
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
    # The 2x2m "table" plate approximates the real floor/work surface as a
    # thin slab perpendicular to whichever base axis is physically vertical,
    # positioned config.piper_mount_height_m below the base_link origin
    # along that axis (measured on the actual rig -- see PIPER_MOUNT_HEIGHT_M
    # in config.py). For a horizontal (floor) mount base_link sits at table
    # height, so that axis is base +Z and the offset is ~0. For a side mount
    # (right/left) the arm is raised on a bracket, so the vertical axis is
    # base +Y instead (test_mit_controller2.py's _room_delta_to_base()
    # empirically verified this for "right" by moving each base axis in
    # isolation on hardware) and the offset is the real, measured mount
    # height -- reusing the floor-mount's near-zero offset here previously
    # placed a 2m collision wall a few cm from the robot's own body and
    # caused a violent collision-avoidance lurch into the floor the instant
    # MIT mode engaged. Do not shrink this offset without re-measuring the
    # rig; a too-small value recreates that hazard.
    key = (config.piper_installation_pos or "horizontal").strip().lower()
    height = max(0.0, float(config.piper_mount_height_m))
    if key == "right":
        table_cuboid = {
            "name": "table",
            "dims": [2.0, 0.05, 2.0],
            "pose": [0.0, -(height + 0.025), 0.0, 1.0, 0.0, 0.0, 0.0],
        }
    elif key == "left":
        # _room_delta_to_base() marks "left" as hardware-unverified (best
        # effort: base -Y vertical, mirrored sign from "right"). Flag it
        # loudly rather than silently trusting an unverified guess.
        print(
            "[curobo_worker] WARNING: 'left' installation's vertical-axis "
            "mapping has not been hardware-verified (see "
            "test_mit_controller2.py _room_delta_to_base) -- the table "
            "collision cuboid pose below is a best-effort guess, not a "
            "measured value."
        )
        table_cuboid = {
            "name": "table",
            "dims": [2.0, 0.05, 2.0],
            "pose": [0.0, height + 0.025, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
    else:
        table_cuboid = {
            "name": "table",
            "dims": [2.0, 2.0, 0.05],
            "pose": [0.0, 0.0, -(height + 0.025), 1.0, 0.0, 0.0, 0.0],
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


def _clamp_joint_trajectory(
    pos: np.ndarray,
    vel: np.ndarray,
    q_current: list[float],
    max_vel_rad_s: float,
    output_hz: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Hard per-joint velocity ceiling, applied to every plan() output.

    Safety backstop against MPC/IK solution-branch-flip spikes: observed on
    hardware right after mpc.reset_robot() (wrist-calibration completion),
    where the EEF target barely moved but consecutive replans picked wildly
    different joint solutions -- one clean case measured a transient J4
    velocity of ~10 rad/s, a real felt jolt. Walks the position trajectory
    cumulatively from q_current so a bad solve can't sneak through as a
    single large position jump even if only one waypoint gets sampled from
    it; velocity is clipped to the same magnitude so v_des stays consistent
    with the clamped positions. Not a tuning knob -- CUROBO_MAX_JOINT_VEL_RAD_S
    should stay well above normal ~0.2-0.4 rad/s operation and only ever bite
    on outliers.
    """
    max_step = max_vel_rad_s / output_hz
    clamped_pos = pos.copy()
    prev = np.asarray(q_current, dtype=pos.dtype)
    triggered = False
    for i in range(len(clamped_pos)):
        delta = clamped_pos[i] - prev
        clipped = np.clip(delta, -max_step, max_step)
        if np.any(clipped != delta):
            triggered = True
        clamped_pos[i] = prev + clipped
        prev = clamped_pos[i]
    clamped_vel = np.clip(vel, -max_vel_rad_s, max_vel_rad_s)
    return clamped_pos, clamped_vel, triggered


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

    def plan(q_current: list[float], target_eef: list[float], reset: bool = False):
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

    return plan, False


# Minimum EEF position movement (m) before a new teleop target is actually
# pushed into the MPC's goal (see _create_v2_mpc_planner docstring).
_GOAL_UPDATE_THRESHOLD_M = 0.005


def _create_v2_mpc_planner(config: CollectionConfig):
    """Closed-loop MPC planner using curobo.model_predictive_control.ModelPredictiveControl.

    Unlike MotionGen/MotionPlanner (plan once per new EEF target), MPC must be
    re-optimized every tick using the live joint state as feedback -- that
    warm-started, receding-horizon re-solve is what makes it MPC. The goal
    pose itself is only updated when the EEF target actually changes.

    Committing every raw target_eef update (25 Hz teleop tracking) to
    update_goal_tool_poses() defeats that warm-starting: the built-in IK
    uses a single seed that's only expected to converge by refining over
    several *consecutive* ticks against the *same* goal (see the
    use_best_effort_ik comment below). If the goal itself changes every
    tick, the seed never gets that many ticks and the arm was observed to
    stall completely a few cm short of the target (measured joint velocity
    pinned at exactly 0 while the MPC kept computing a small nonzero
    v_des -- reproduced and root-caused in test_mit_controller2.py). Fix:
    only commit a new goal once the raw target has moved more than
    _GOAL_UPDATE_THRESHOLD_M since the last committed goal.
    """
    import os

    import torch
    from curobo.content import get_task_configs_path
    from curobo.model_predictive_control import (
        ModelPredictiveControl,
        ModelPredictiveControlCfg,
    )
    from curobo.types import DeviceCfg, GoalToolPose, JointState
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
    from curobo._src.types.robot import RobotCfg
    from curobo._src.util.config_io import load_yaml

    device_cfg = DeviceCfg()
    kinematics_cfg = KinematicsCfg.from_robot_yaml_file(
        str(config.curobo_xrdf_path),
        tool_frames=["flange_link"],
        device_cfg=device_cfg,
        urdf_path=str(config.curobo_urdf_path),
    )
    robot_cfg = RobotCfg(kinematics=kinematics_cfg, device_cfg=device_cfg)
    all_cuboids = _world_cuboids(config)

    # lbfgs_mpc.yml's default tool_pose_cfg.weight is [position=5000, orientation=200]
    # -- a 25:1 ratio that lets the real-time optimizer sacrifice orientation
    # tracking to chase a fast-moving teleop position target, plus the default
    # orientation_tolerance=0.05rad (~2.9deg) lets the goal-seeding IK accept a
    # coarse orientation match up front. Observed on hardware: EEF tilts even
    # at rest right after a goal is set, worsening during motion. Tighten both.
    mpc_cost_yaml = os.path.join(str(get_task_configs_path()), "mpc", "lbfgs_mpc.yml")
    mpc_cost_cfg = load_yaml(mpc_cost_yaml)
    mpc_cost_cfg["rollout"]["cost_cfg"]["tool_pose_cfg"]["weight"][1] = 2000.0

    output_hz = config.curobo_output_hz
    interpolation_steps = 4  # fixed by cuRobo; command_dt = optimization_dt / interpolation_steps
    mpc_cfg = ModelPredictiveControlCfg.create(
        robot=robot_cfg,
        optimizer_configs=[mpc_cost_cfg],
        scene_model=_v2_scene_model(all_cuboids),
        collision_cache={"cuboid": max(1, len(all_cuboids))},
        device_cfg=device_cfg,
        self_collision_check=True,
        use_cuda_graph=True,
        optimization_dt=interpolation_steps / output_hz,
        interpolation_steps=interpolation_steps,
        orientation_tolerance=0.01,
    )
    mpc = ModelPredictiveControl(mpc_cfg)
    print("[curobo_worker] Using cuRobo v2 ModelPredictiveControl (MPC).")

    # optimize_action_sequence() returns get_command_sequence(), a slice of
    # the interpolated horizon of length (command_end_idx - command_start_idx).
    # The library defaults command_end_idx to interpolation_steps * 2, which
    # (with command_start_idx == interpolation_steps for bspline transition
    # models) yields exactly `interpolation_steps` waypoints -- here 4
    # waypoints * 1/output_hz = 33.3ms of trajectory. But this planner is
    # only re-invoked every 1/curobo_input_hz = 40ms, so the pushed
    # trajectory is ~7ms short of covering the gap to the next replan on
    # every single cycle (worse once GPU solve latency is added). Each
    # shortfall starves CuRoboTrajectoryBuffer, latching hold_pos in
    # piper_worker._run_mit_step() and causing a stall/resume judder at
    # curobo_input_hz. Widen the exposed command window so it comfortably
    # covers one input period with margin.
    output_dt_s = 1.0 / output_hz
    input_period_s = 1.0 / config.curobo_input_hz
    min_waypoints = math.ceil(input_period_s / output_dt_s)
    command_span = min_waypoints + interpolation_steps  # +1 window of margin
    tem = mpc.trajectory_execution_manager
    tem.command_end_idx = tem.command_start_idx + command_span
    print(
        f"[curobo_worker] MPC command window widened to {command_span} waypoints "
        f"({command_span * output_dt_s * 1000:.1f}ms) to cover the "
        f"{input_period_s * 1000:.1f}ms replan period."
    )

    tool_frame = mpc.tool_frames[0]
    setup_done = False
    last_goal_eef: list[float] | None = None
    last_velocity: "torch.Tensor | None" = None
    last_acceleration: "torch.Tensor | None" = None

    def _current_state(q_current: list[float], *, reset_velocity: bool) -> "JointState":
        nonlocal last_velocity, last_acceleration
        q_t = torch.tensor([q_current], dtype=device_cfg.dtype, device=device_cfg.device)
        state = JointState.from_position(q_t, joint_names=mpc.joint_names)
        # MPC's warm-started B-spline transition model expects continuity
        # with the velocity/acceleration it commanded last tick. Re-zeroing
        # both every call (JointState.from_position's default) tells the
        # optimizer "you're at rest" even mid-trajectory, which stalls
        # convergence almost completely (verified: with this fix a nearby
        # goal converges in ~2s; without it, error stays put indefinitely).
        if not reset_velocity and last_velocity is not None:
            state.velocity = last_velocity
            state.acceleration = last_acceleration
        return state

    def plan(q_current: list[float], target_eef: list[float], reset: bool = False):
        nonlocal setup_done, last_goal_eef, last_velocity, last_acceleration
        current_state = _current_state(q_current, reset_velocity=(not setup_done) or reset)

        if not setup_done:
            print("[curobo_worker] Warming up MPC (initial setup + cold-start solve)...")
            mpc.setup(current_state)
            setup_done = True
            last_goal_eef = None
            print("[curobo_worker] Warmup complete.")
        elif reset:
            mpc.reset_robot(current_state)

        needs_goal_update = last_goal_eef is None or math.sqrt(
            sum((target_eef[i] - last_goal_eef[i]) ** 2 for i in range(3))
        ) >= _GOAL_UPDATE_THRESHOLD_M
        if needs_goal_update:
            x, y, z, roll, pitch, yaw = target_eef
            w_q, x_q, y_q, z_q = _euler_xyz_to_quat_wxyz(roll, pitch, yaw)
            goal = GoalToolPose(
                tool_frames=[tool_frame],
                position=torch.tensor(
                    [[[[[x, y, z]]]]], dtype=device_cfg.dtype, device=device_cfg.device
                ),
                quaternion=torch.tensor(
                    [[[[[w_q, x_q, y_q, z_q]]]]], dtype=device_cfg.dtype, device=device_cfg.device
                ),
            )
            # use_best_effort_ik: the MPC's built-in IK solver uses a single
            # seed and often fails to converge to tight tolerance even for
            # already-reachable poses (observed even for the arm's own
            # current pose). MPC continuously re-optimizes toward the goal
            # every tick regardless, so an inexact IK seed is refined away
            # over subsequent ticks -- unlike MotionGen, which plans once and
            # needs the goal nailed down up front.
            if mpc.update_goal_tool_poses(goal, use_best_effort_ik=True):
                last_goal_eef = target_eef
            else:
                print(f"[curobo_worker] MPC goal IK failed for tgt_eef={target_eef}")

        result = mpc.optimize_action_sequence(current_state)
        if result.action_sequence is None:
            last_velocity = None
            last_acceleration = None
            return None
        last_velocity = result.action_sequence.velocity[:, -1, :].detach()
        last_acceleration = result.action_sequence.acceleration[:, -1, :].detach()
        return _joint_state_to_arrays(result.action_sequence)

    return plan, True


def _create_remote_planner(config: CollectionConfig):
    """Offload the MPC solve to a second machine's GPU over TCP.

    Local solves on this box's GPU were measured at 70-150ms/call while
    nvidia-smi showed the GPU itself mostly idle (1-40% util) -- suspected
    GIL contention from the other CPU-bound worker threads sharing this
    process (see CUROBO_REMOTE_ENABLED in config.py). The remote box runs
    scripts/curobo_solver_server.py, which calls the *same*
    _create_curobo_planner() locally and serves plan() over a plain framed
    TCP socket (scripts/piper_api/curobo_remote_protocol.py). Matches the
    plan(q_current, target_eef, reset) -> (pos, vel) | None contract exactly,
    so run_curobo_worker()'s loop needs no changes for this backend.
    """
    from scripts.piper_api import curobo_remote_protocol as protocol

    host = config.curobo_remote_host
    port = config.curobo_remote_port
    timeout_s = config.curobo_remote_timeout_s
    sock: "object | None" = None

    def _connect():
        import socket as socket_module

        s = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
        protocol.set_low_latency(s)
        s.settimeout(timeout_s)
        print(f"[curobo_worker] Connecting to remote solver {host}:{port} ...")
        s.connect((host, port))
        print("[curobo_worker] Connected to remote solver.")
        return s

    sock = _connect()
    _rtt_diag_next_print = 0.0

    def plan(q_current: list[float], target_eef: list[float], reset: bool = False):
        nonlocal sock, _rtt_diag_next_print
        _rtt_start = time.monotonic()
        try:
            protocol.send_plan_request(sock, q_current, target_eef, reset)
            result = protocol.recv_plan_response(sock)
        except Exception as exc:
            rtt_ms = (time.monotonic() - _rtt_start) * 1000
            print(
                f"[curobo_worker] remote solver connection error after {rtt_ms:.1f}ms: "
                f"{exc} -- reconnecting"
            )
            try:
                sock.close()
            except Exception:
                pass
            try:
                sock = _connect()
            except Exception as reconnect_exc:
                print(f"[curobo_worker] remote solver reconnect failed: {reconnect_exc}")
            return None
        rtt_ms = (time.monotonic() - _rtt_start) * 1000
        _now_rtt_diag = time.monotonic()
        if result is None or _now_rtt_diag >= _rtt_diag_next_print:
            _rtt_diag_next_print = _now_rtt_diag + 0.2  # 5 Hz (always print failures)
            print(f"[diag/remote_rtt] {rtt_ms:.1f}ms  ok={result is not None}")
        if result is None:
            return None
        return result

    return plan, True


def _create_curobo_planner(config: CollectionConfig):
    if config.curobo_remote_enabled:
        return _create_remote_planner(config)
    if _module_available("curobo.geom.sdf.world"):
        return _create_v1_planner(config)
    if _module_available("curobo.model_predictive_control"):
        return _create_v2_mpc_planner(config)
    raise RuntimeError(
        "Installed curobo package exposes neither v1 MotionGen "
        "(curobo.geom.sdf.world) nor v2 ModelPredictiveControl "
        "(curobo.model_predictive_control)."
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
    GPU thread: polls SharedState for the EEF target at CUROBO_INPUT_HZ and
    pushes a joint trajectory (sampled at CUROBO_OUTPUT_HZ) into
    trajectory_buffer. For the event-driven v1 MotionGen backend this only
    (re)plans when the target changes; for the closed-loop v2 MPC backend
    this re-optimizes every tick using the live joint state as feedback.
    """
    plan, is_continuous = _create_curobo_planner(config)

    init_done.set()

    input_period_s = 1.0 / config.curobo_input_hz
    last_eef: list[float] | None = None
    needs_reseed = False
    _diag_plan_count = 0
    # Per-joint judder diagnostic: distinguish "robot is physically
    # oscillating" (measured q jumps tick to tick -- a gain/control issue)
    # from "the plan itself is unstable" (this call's final commanded angle
    # differs a lot from the previous call's, even though the robot barely
    # moved -- an MPC/IK solution-branch-flip issue, not a gain issue).
    # Uncapped (unlike the first-5 print above) but self-limiting: only
    # fires past the threshold. Note this only samples once per cuRobo
    # replan (curobo_input_hz, 25Hz -> Nyquist ~12.5Hz) -- it cannot resolve
    # oscillation faster than that; see piper_worker.JointJitterDiag (120Hz)
    # for the higher-frequency band this can alias away.
    _JOINT_DIAG_THRESHOLD_RAD = 0.01  # ~0.57 deg per curobo_input_hz tick (40ms)
    _prev_q_measured: list[float] | None = None
    _prev_q_planned_end: list[float] | None = None
    _solve_diag_next_print = 0.0

    while not shared.shutdown.is_set():
        loop_start = time.monotonic()

        if not shared.teleop_enabled() or shared.backdrive_mode():
            # backdrive_mode also keeps teleop_enabled True (see runner.py's
            # 'z' handler), but target_eef is frozen at whatever it was right
            # before backdrive started while the operator moves the arm by
            # hand -- if we kept replanning here, the buffer would stay full
            # of fresh (non-stale) waypoints aimed at that stale target, and
            # exiting backdrive would skip the "hold current position"
            # fallback in piper_worker._run_mit_step() and snap the arm back
            # toward it. Pausing here lets the buffer go stale during
            # backdrive so that fallback engages instead.
            trajectory_buffer.reset()
            needs_reseed = True
            elapsed = time.monotonic() - loop_start
            shared.shutdown.wait(max(0.0, input_period_s - elapsed))
            continue

        target_eef = shared.latest_target_eef()
        if target_eef is None:
            elapsed = time.monotonic() - loop_start
            shared.shutdown.wait(max(0.0, input_period_s - elapsed))
            continue
        if not is_continuous and target_eef == last_eef:
            elapsed = time.monotonic() - loop_start
            shared.shutdown.wait(max(0.0, input_period_s - elapsed))
            continue

        q_current = shared.latest_joint_positions()
        if q_current is None:
            elapsed = time.monotonic() - loop_start
            shared.shutdown.wait(max(0.0, input_period_s - elapsed))
            continue

        last_eef = target_eef
        solve_start = time.monotonic()

        try:
            planned = plan(q_current, target_eef, reset=needs_reseed)
            needs_reseed = False
            if planned is not None:
                pos, vel = planned
                pos, vel, clamp_triggered = _clamp_joint_trajectory(
                    pos, vel, q_current, config.curobo_max_joint_vel_rad_s, config.curobo_output_hz
                )
                if clamp_triggered:
                    print(
                        "[curobo_worker] WARNING: joint velocity clamp engaged "
                        f"(cap={config.curobo_max_joint_vel_rad_s} rad/s) -- plan output "
                        "exceeded the safety ceiling and was capped."
                    )
                # pos[0] was computed for q_current as sampled at the top of
                # this iteration, but the GPU solve above takes real wall-clock
                # time, and push_trajectory() always resets the consumer's
                # read_idx to 0. In principle this correction drops however
                # many waypoints the solve itself consumed so the buffer picks
                # up close to "now" instead of "then" -- but on this hardware
                # measured solves regularly run 70-150ms (vs. the 40ms input
                # period the buffer coverage was sized for), so the naive
                # correction was skipping 8 of 9 waypoints almost every cycle,
                # leaving just 1 (8.3ms) to cover a 70-150ms gap -- far worse
                # starvation than not correcting at all. Cap the skip at half
                # the buffer so a slow solve degrades to "a bit stale" instead
                # of "buffer nearly empty".
                solve_elapsed_s = time.monotonic() - solve_start
                skip = int(round(solve_elapsed_s * config.curobo_output_hz))
                skip = max(0, min(skip, len(pos) // 2))
                if skip > 0:
                    pos = pos[skip:]
                    vel = vel[skip:]
                _now_solve_diag = time.monotonic()
                if _now_solve_diag >= _solve_diag_next_print:
                    _solve_diag_next_print = _now_solve_diag + 1.0  # 1 Hz
                    print(
                        f"[diag/solve] solve={solve_elapsed_s * 1000:.1f}ms  "
                        f"skip={skip}/{skip + len(pos)} waypoints "
                        f"({skip / config.curobo_output_hz * 1000:.1f}ms dropped)"
                    )
                _diag_plan_count += 1
                if _diag_plan_count <= 5:
                    print(
                        f"[diag/cuRobo #{_diag_plan_count}] Plan OK  "
                        f"tgt_eef={[f'{v:+.3f}' for v in target_eef]}  "
                        f"waypoints={len(pos)}  "
                        f"solve={solve_elapsed_s * 1000:.1f}ms  skip={skip}  "
                        f"q_start={[f'{v:+.3f}' for v in pos[0].tolist()]}  "
                        f"q_end={[f'{v:+.3f}' for v in pos[-1].tolist()]}"
                    )
                trajectory_buffer.push_trajectory(pos, vel, dt_s=1.0 / config.curobo_output_hz)

                q_measured = list(q_current)
                q_planned_end = [float(x) for x in pos[-1].tolist()]
                measured_delta = (
                    None if _prev_q_measured is None
                    else [q_measured[j] - _prev_q_measured[j] for j in range(6)]
                )
                planned_delta = (
                    None if _prev_q_planned_end is None
                    else [q_planned_end[j] - _prev_q_planned_end[j] for j in range(6)]
                )
                flagged = [
                    j for j in range(6)
                    if (measured_delta is not None and abs(measured_delta[j]) > _JOINT_DIAG_THRESHOLD_RAD)
                    or (planned_delta is not None and abs(planned_delta[j]) > _JOINT_DIAG_THRESHOLD_RAD)
                ]
                if flagged:
                    def _fmt(deltas: "list[float] | None", j: int) -> str:
                        return "n/a" if deltas is None else f"{deltas[j]:+.4f}"

                    print(
                        "[diag/joint] "
                        + "  ".join(
                            f"J{j + 1}(measured_dq={_fmt(measured_delta, j)},"
                            f"planned_end_dq={_fmt(planned_delta, j)})"
                            for j in flagged
                        )
                    )
                _prev_q_measured = q_measured
                _prev_q_planned_end = q_planned_end
            else:
                print(
                    f"[diag/cuRobo] Plan FAILED  tgt_eef={[f'{v:+.3f}' for v in target_eef]}  "
                    f"q_current={[f'{v:+.3f}' for v in q_current]}"
                )
        except Exception as exc:
            print(f"[curobo_worker] plan error: {exc}")

        elapsed = time.monotonic() - loop_start
        shared.shutdown.wait(max(0.0, input_period_s - elapsed))
