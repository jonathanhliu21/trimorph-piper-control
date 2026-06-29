from __future__ import annotations

import time
from typing import TYPE_CHECKING

from delta_teleop_controller import DeltaTeleopController, load_delta_teleop_config
from scripts.piper_api.teacher_mode import get_teacher_mode

from .config import RAW_EEF_FIELDS, CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .shared_state import SharedState

if TYPE_CHECKING:
    from scripts.piper_api.curobo_trajectory_buffer import CuRoboTrajectoryBuffer
    from scripts.piper_api.gravity_compensation import MujocoGravityCompensator
    from scripts.piper_api.piper_api import PiperAPI


def _run_mit_step(
    api: "PiperAPI",
    grav_comp: "MujocoGravityCompensator",
    shared: SharedState,
    buf: "CuRoboTrajectoryBuffer",
    config: CollectionConfig,
) -> None:
    """Single 120 Hz MIT impedance step: gravity feedforward + PD tracking."""
    joint_state = api.get_joint_state()
    if joint_state is None:
        api.move_mit_torques([0.0] * 6, kp=0.0, kd=0.0, clamp=True)
        return

    q = joint_state.joints

    motor_state = api.get_motor_state()
    v = motor_state.velocities if motor_state is not None else [0.0] * 6

    # Publish for cuRobo replanning
    shared.set_latest_joint_positions(q)

    waypoint = buf.next_waypoint()

    grav_result = grav_comp.compute(
        q,
        joint_velocities=v,
        damping=config.curobo_grav_comp_damping,
    )
    grav_torques = grav_result.command_torques

    if waypoint is None:
        # Buffer exhausted → hold current position with gravity compensation
        p_des = list(q)
        v_des = [0.0] * 6
    else:
        p_des = waypoint.positions.tolist()
        v_des = waypoint.velocities.tolist()

    api.move_mit_torques(
        grav_torques,
        p_des=p_des,
        v_des=v_des,
        kp=config.curobo_mit_kp,
        kd=config.curobo_mit_kd,
        clamp=True,
    )


def run_piper_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    trajectory_buffer: "CuRoboTrajectoryBuffer | None" = None,
    *,
    init_done,
) -> None:
    from scripts.piper_api.piper_api import PiperAPI

    api = PiperAPI(
        firmware=config.piper_firmware,
        interface=config.piper_interface,
        channel=config.piper_channel,
        installation_pos=config.piper_installation_pos,
        payload=config.piper_payload,
    )
    api.connect()
    print("Piper connected:", api.is_ok(), "receive FPS:", api.get_fps())

    grav_comp = None
    if config.curobo_enabled:
        from scripts.piper_api.gravity_compensation import MujocoGravityCompensator
        grav_comp = MujocoGravityCompensator(
            driver=config.piper_firmware,
            installation_pos=config.piper_installation_pos,
        )
        api.enable()
        api.set_mit_mode()
        print("[piper_worker] cuRobo MIT mode enabled.")

    init_done.set()

    sample_period_ns = int(round(1_000_000_000 / config.piper_rate_hz))
    command_period_ns = int(round(1_000_000_000 / config.piper_command_rate_hz))
    teacher_period_ns = int(round(1_000_000_000 / config.teacher_mode_poll_hz))
    mit_period_ns = (
        int(round(1_000_000_000 / config.curobo_output_hz))
        if config.curobo_enabled else 0
    )

    now_ns = time.monotonic_ns()
    next_sample_ns = now_ns
    next_command_ns = now_ns
    next_teacher_ns = 0
    next_mit_ns = now_ns

    sample_idx = 0
    sample_session: EpisodeBuffer | None = None
    delta_config = load_delta_teleop_config(config.delta_teleop_config_path)
    delta_controller = DeltaTeleopController(delta_config)
    reported_calibrated = False
    last_calibration_report = -1
    arm_teleop_active = False
    print(
        "Delta wrist teleop loaded:",
        f"config={config.delta_teleop_config_path}",
        f"orientation_enabled={delta_config.orientation_enabled}",
        f"scale_factor={delta_config.scale_factor}",
    )
    try:
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()
            if now_ns >= next_teacher_ns:
                active, status = get_teacher_mode(api)
                shared.set_teacher_mode(active, status)
                next_teacher_ns = now_ns + teacher_period_ns

            if now_ns >= next_command_ns:
                teacher_active, _ = shared.teacher_mode()
                should_run_arm_teleop = (not teacher_active) and shared.teleop_enabled()
                if should_run_arm_teleop and not arm_teleop_active:
                    delta_controller.reset_calibration()
                    reported_calibrated = False
                    last_calibration_report = -1
                    print("Arm teleop enabled; recalibrating wrist reference.")
                arm_teleop_active = should_run_arm_teleop
                if should_run_arm_teleop:
                    wrist_pose_cam = shared.latest_wrist_pose_cam()
                    robot_eef = None if delta_controller.calibrated else api.get_eef_pose()
                    target_eef = delta_controller.update(wrist_pose_cam, robot_eef)
                    status = delta_controller.status()
                    if not status.calibrated and status.calibration_count != last_calibration_report:
                        last_calibration_report = status.calibration_count
                        print(
                            "Wrist delta calibration:",
                            f"{status.calibration_count}/{status.calibration_frames}",
                        )
                    if status.calibrated and not reported_calibrated:
                        reported_calibrated = True
                        print("Wrist delta calibration complete.")
                    if target_eef is not None:
                        if config.curobo_enabled:
                            # cuRobo path: publish target for GPU worker to plan
                            shared.set_target_eef(target_eef)
                        else:
                            # Legacy path: firmware IK at 30 Hz
                            api.move_eef(target_eef)
                next_command_ns += command_period_ns
                while next_command_ns < now_ns:
                    next_command_ns += command_period_ns

            # 120 Hz MIT impedance loop (cuRobo path only)
            if config.curobo_enabled and trajectory_buffer is not None and now_ns >= next_mit_ns:
                assert grav_comp is not None
                _run_mit_step(api, grav_comp, shared, trajectory_buffer, config)
                next_mit_ns += mit_period_ns
                while next_mit_ns < now_ns:
                    next_mit_ns += mit_period_ns

            if now_ns >= next_sample_ns:
                eef = api.get_eef_state()
                sample_ns = time.monotonic_ns()
                session = recorder.current()
                if session is not None and eef is not None:
                    if session is not sample_session:
                        sample_session = session
                        sample_idx = 0
                    row = {
                        "sample_idx": sample_idx,
                        "monotonic_ns": sample_ns,
                        "wall_time_s": f"{time.time():.6f}",
                        "elapsed_s": f"{session.elapsed_s(sample_ns):.9f}",
                        "robot_timestamp_s": f"{eef.timestamp:.6f}",
                        "robot_hz": f"{eef.hz:.3f}",
                    }
                    row.update(
                        {
                            name: f"{float(value):.9f}"
                            for name, value in zip(RAW_EEF_FIELDS, eef.pose)
                        }
                    )
                    session.append_piper(row)
                    sample_idx += 1
                next_sample_ns += sample_period_ns
                while next_sample_ns < sample_ns:
                    next_sample_ns += sample_period_ns
            else:
                # When cuRobo MIT loop is active, sleep at most one MIT period
                if config.curobo_enabled:
                    next_wake_ns = min(next_sample_ns, next_mit_ns, next_command_ns)
                else:
                    next_wake_ns = next_sample_ns
                wait_s = (next_wake_ns - time.monotonic_ns()) / 1e9
                if wait_s > 0:
                    shared.shutdown.wait(min(wait_s, 0.005))
    finally:
        api.disconnect()
