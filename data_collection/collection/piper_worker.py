from __future__ import annotations

import time

from delta_teleop_controller import DeltaTeleopController, load_delta_teleop_config
from scripts.piper_api.teacher_mode import get_teacher_mode

from .config import RAW_EEF_FIELDS, CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .shared_state import SharedState

def run_piper_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
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
    init_done.set()

    sample_period_ns = int(round(1_000_000_000 / config.piper_rate_hz))
    command_period_ns = int(round(1_000_000_000 / config.piper_command_rate_hz))
    teacher_period_ns = int(round(1_000_000_000 / config.teacher_mode_poll_hz))
    next_sample_ns = time.monotonic_ns()
    next_command_ns = next_sample_ns
    next_teacher_ns = 0
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
                        api.move_eef(target_eef)
                next_command_ns += command_period_ns
                while next_command_ns < now_ns:
                    next_command_ns += command_period_ns

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
                shared.shutdown.wait(min((next_sample_ns - now_ns) / 1e9, 0.005))
    finally:
        api.disconnect()
