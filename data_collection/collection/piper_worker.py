from __future__ import annotations

import math
import time

import numpy as np

from scripts.piper_api.teacher_mode import get_teacher_mode

from .config import RAW_EEF_FIELDS, CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .shared_state import SharedState


def _matrix_to_euler_xyz(r: np.ndarray) -> tuple[float, float, float]:
    # fixed-axis XYZ: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    pitch = math.asin(float(-r[2, 0]))
    cp = math.cos(pitch)
    if abs(cp) < 1e-9:
        roll = 0.0
        yaw = math.atan2(float(-r[0, 1]), float(r[1, 1]))
    else:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return roll, pitch, yaw


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
    wrist_cam_to_piper = np.asarray(config.wrist_cam_to_piper_eef_4x4, dtype=np.float64)
    try:
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()
            if now_ns >= next_teacher_ns:
                active, status = get_teacher_mode(api)
                shared.set_teacher_mode(active, status)
                next_teacher_ns = now_ns + teacher_period_ns

            if now_ns >= next_command_ns:
                if shared.teleop_enabled():
                    wrist_pose_cam = shared.latest_wrist_pose_cam()
                    if wrist_pose_cam is not None and wrist_pose_cam.shape == (4, 4):
                        target = wrist_cam_to_piper @ wrist_pose_cam
                        r = target[:3, :3]
                        t = target[:3, 3]
                        roll, pitch, yaw = _matrix_to_euler_xyz(r)
                        api.move_eef([float(t[0]), float(t[1]), float(t[2]), float(roll), float(pitch), float(yaw)])
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
