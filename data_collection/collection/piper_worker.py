from __future__ import annotations

import time
from typing import TYPE_CHECKING

from delta_teleop_controller import DeltaTeleopController, load_delta_teleop_config
from scripts.piper_api.teacher_mode import get_teacher_mode

from .config import RAW_EEF_FIELDS, CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .shared_state import SharedState

from scripts.piper_api.gravity_compensation import MIDAS_HAND_MODEL_PATH

# Home EEF pose [x, y, z, roll, pitch, yaw] in metres / radians — from teleop_arm.py HOME_POSE
_HOME_EEF = [0.325, -0.3, 0.2, -1.5707963267948966, -0.659612, -1.5707963267948966]

# Backdrive mode parameters — matched to:
#   test_mit_controller.py --firmware v183 --torque-scale 1.05 --coulomb-comp 0.05 --viscous-comp 0.05
_BACKDRIVE_RATE_HZ = 120.0
_BACKDRIVE_TORQUE_SCALE = 1.05
_BACKDRIVE_COULOMB_COMP = 0.05
_BACKDRIVE_VISCOUS_COMP = 0.05

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
    hold_pos: "list[float] | None",
) -> "list[float] | None":
    """Single 120 Hz MIT impedance step: gravity feedforward + PD tracking.

    hold_pos is latched the first time the trajectory buffer runs dry and kept
    fixed until a new trajectory arrives.  This ensures kp*(p_des - q) is
    non-zero and actually corrects residual drift instead of being zeroed out
    by a continuously-updated p_des = q.

    Returns the (possibly updated) hold_pos.
    """
    if not api.is_ok():
        return hold_pos
    joint_state = api.get_joint_state()
    if joint_state is None:
        return hold_pos

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
        torque_scale=config.curobo_grav_comp_torque_scale,
    )
    grav_torques = grav_result.command_torques

    if waypoint is None:
        if hold_pos is None:
            hold_pos = list(q)  # latch once; keep fixed until trajectory resumes
        p_des = hold_pos
        v_des = [0.0] * 6
    else:
        hold_pos = None  # clear latch while trajectory is active
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
    return hold_pos


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
    if not api.is_ok():
        raise RuntimeError(
            "Piper arm is not responding on the CAN bus (receive FPS=0). "
            "Check that the arm is powered on and the CAN cable is connected."
        )

    from scripts.piper_api.gravity_compensation import MujocoGravityCompensator
    firmware_version = api.get_software_version()
    print(f"[piper_worker] firmware version: {firmware_version!r}")
    grav_comp = MujocoGravityCompensator(
        model_path=MIDAS_HAND_MODEL_PATH,
        driver=config.piper_firmware,
        firmware_version=firmware_version,
        installation_pos=config.piper_installation_pos,
    )
    # api.enable() and api.set_mit_mode() are deferred until 'e' or 'b' is first pressed
    # so the arm stays in its current mode until the operator activates teleop/backdrive.

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
    mit_mode_active = False
    mit_hold_pos: list[float] | None = None
    backdrive_active = False
    backdrive_period_ns = int(round(1_000_000_000 / _BACKDRIVE_RATE_HZ))
    next_backdrive_ns = now_ns
    _mit_diag_until: float = 0.0
    _mit_diag_next_print: float = 0.0
    mit_stale_since: float | None = None
    mit_stale_warned = False
    delta_config = load_delta_teleop_config(config.delta_teleop_config_path)
    delta_controller = DeltaTeleopController(delta_config)
    reported_calibrated = False
    last_calibration_report = -1
    arm_teleop_active = False
    was_home_requested = False
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

            # --- Backdrive mode state transitions ---
            backdrive_requested = shared.backdrive_mode()
            if backdrive_requested and not backdrive_active:
                if not mit_mode_active:
                    api.enable()
                    api.set_mit_mode()
                    mit_mode_active = True
                backdrive_active = True
                backdrive_hold_pos = None
                print(
                    f"[piper_worker] Backdrive mode ON "
                    f"(torque_scale={_BACKDRIVE_TORQUE_SCALE}, "
                    f"coulomb={_BACKDRIVE_COULOMB_COMP}, viscous={_BACKDRIVE_VISCOUS_COMP})"
                )
            elif backdrive_active and not backdrive_requested:
                backdrive_active = False
                mit_hold_pos = None  # re-latch curobo hold pos
                if not config.curobo_enabled and mit_mode_active:
                    # MIT 모드에서 빠져나와 position 모드로 복귀해야 move_eef()가 작동함
                    api.disable()
                    api.enable()
                    mit_mode_active = False
                    print("[piper_worker] Backdrive mode OFF — arm re-enabled in position mode.")
                else:
                    print("[piper_worker] Backdrive mode OFF.")

            if now_ns >= next_command_ns:
                teacher_active, _ = shared.teacher_mode()
                should_run_arm_teleop = (not teacher_active) and shared.teleop_enabled() and not backdrive_active
                if should_run_arm_teleop and not arm_teleop_active:
                    delta_controller.reset_calibration()
                    reported_calibrated = False
                    last_calibration_report = -1
                    print("Arm teleop enabled; recalibrating wrist reference.")
                    if config.curobo_enabled and not mit_mode_active:
                        q_pre = api.get_joint_angles()
                        eef_pre = api.get_eef_pose()
                        print(
                            f"[diag] Before enable — q={[f'{v:+.4f}' for v in (q_pre or [])]}  "
                            f"eef={[f'{v:+.4f}' for v in (eef_pre or [])]}"
                        )
                        print(
                            f"[diag] grav_comp model={grav_comp.model_path.name}  "
                            f"gravity={list(grav_comp.get_gravity())}  "
                            f"output_scales={grav_comp.output_scales}"
                        )
                        api.enable()
                        api.set_mit_mode()
                        mit_mode_active = True
                        _mit_diag_until = time.monotonic() + 3.0
                        q_post = api.get_joint_angles()
                        eef_post = api.get_eef_pose()
                        print(
                            f"[diag] After MIT mode — q={[f'{v:+.4f}' for v in (q_post or [])]}  "
                            f"eef={[f'{v:+.4f}' for v in (eef_post or [])]}"
                        )
                        print("[piper_worker] cuRobo MIT mode enabled.")
                if arm_teleop_active and not should_run_arm_teleop:
                    mit_hold_pos = None  # re-latch on next MIT cycle after teleop resumes
                    shared.set_home_requested(False)
                arm_teleop_active = should_run_arm_teleop
                if should_run_arm_teleop:
                    home_req = shared.home_requested()
                    if was_home_requested and not home_req:
                        # Home mode just cancelled: recalibrate wrist reference from new position
                        delta_controller.reset_calibration()
                        reported_calibrated = False
                        last_calibration_report = -1
                        print("Home mode cancelled; recalibrating wrist reference.")
                    was_home_requested = home_req

                    if home_req:
                        if config.curobo_enabled:
                            shared.set_target_eef(list(_HOME_EEF))
                        else:
                            api.move_eef(_HOME_EEF)
                    else:
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

            # Backdrive MIT loop: pure gravity comp, kp=0 kd=0 — arm freely backdriveable
            if mit_mode_active and backdrive_active and now_ns >= next_backdrive_ns:
                if api.is_ok():
                    joint_state = api.get_joint_state()
                    motor_state = api.get_motor_state()
                    if joint_state is not None:
                        q = joint_state.joints
                        v = motor_state.velocities if motor_state is not None else [0.0] * 6
                        shared.set_latest_joint_positions(q)
                        grav_result = grav_comp.compute(
                            q,
                            joint_velocities=v,
                            coulomb_comp=_BACKDRIVE_COULOMB_COMP,
                            viscous_comp=_BACKDRIVE_VISCOUS_COMP,
                            torque_scale=_BACKDRIVE_TORQUE_SCALE,
                        )
                        api.move_mit_torques(
                            grav_result.command_torques,
                            p_des=q,
                            v_des=[0.0] * 6,
                            kp=0.0,
                            kd=0.0,
                            clamp=True,
                        )
                next_backdrive_ns += backdrive_period_ns
                while next_backdrive_ns < now_ns:
                    next_backdrive_ns += backdrive_period_ns

            # 120 Hz MIT impedance loop (cuRobo path only, after MIT mode is entered)
            if config.curobo_enabled and mit_mode_active and trajectory_buffer is not None and not backdrive_active and now_ns >= next_mit_ns:
                assert grav_comp is not None
                _now_diag = time.monotonic()
                if _now_diag < _mit_diag_until and _now_diag >= _mit_diag_next_print:
                    _mit_diag_next_print = _now_diag + 0.5  # print at 2 Hz
                    js = api.get_joint_state()
                    ms = api.get_motor_state()
                    if js is not None:
                        q_diag = js.joints
                        v_diag = ms.velocities if ms is not None else [0.0] * 6
                        gr = grav_comp.compute(
                            q_diag, joint_velocities=v_diag,
                            damping=config.curobo_grav_comp_damping,
                            torque_scale=config.curobo_grav_comp_torque_scale,
                        )
                        tgt = shared.latest_target_eef()
                        print(
                            f"[diag/MIT] q={[f'{v:+.3f}' for v in q_diag]}  "
                            f"grav_τ={[f'{v:+.3f}' for v in gr.command_torques]}  "
                            f"buf_stale={trajectory_buffer.is_stale()}  "
                            f"target_eef={[f'{v:+.3f}' for v in (tgt or [])]}"
                        )
                mit_hold_pos = _run_mit_step(api, grav_comp, shared, trajectory_buffer, config, mit_hold_pos)
                if trajectory_buffer.is_stale():
                    if mit_stale_since is None:
                        mit_stale_since = time.monotonic()
                    elif not mit_stale_warned and (
                        time.monotonic() - mit_stale_since >= config.curobo_stale_timeout_s
                    ):
                        print(
                            f"[piper_worker] WARNING: cuRobo trajectory stale for "
                            f">{config.curobo_stale_timeout_s:.1f}s — replanning may have failed."
                        )
                        mit_stale_warned = True
                else:
                    mit_stale_since = None
                    mit_stale_warned = False
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
                if mit_mode_active and backdrive_active:
                    next_wake_ns = min(next_sample_ns, next_backdrive_ns, next_command_ns)
                elif config.curobo_enabled and mit_mode_active:
                    next_wake_ns = min(next_sample_ns, next_mit_ns, next_command_ns)
                else:
                    next_wake_ns = next_sample_ns
                wait_s = (next_wake_ns - time.monotonic_ns()) / 1e9
                if wait_s > 0:
                    shared.shutdown.wait(min(wait_s, 0.005))
    finally:
        api.disconnect()
