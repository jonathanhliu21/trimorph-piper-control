from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import CANONICAL_MIDAS_MOTOR_IDS, CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .shared_state import SharedState


def load_midas_config(config: CollectionConfig):
    from midas_hand_api import HandConfig

    path = Path(config.midas_hand_config).expanduser()
    if path.exists():
        hand_config = HandConfig.load(path)
    else:
        hand_config = HandConfig.xm335_t323()

    updates: dict[str, object] = {
        "baudrate": config.midas_hand_baudrate,
    }
    if config.midas_hand_port is not None:
        updates["port"] = config.midas_hand_port
    if config.midas_current_limit_ma is not None:
        updates["goal_current_limit"] = int(config.midas_current_limit_ma)
    return replace(hand_config, **updates)


def run_midas_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
) -> None:
    from midas_hand_api import MidasHand

    hand = MidasHand(config=load_midas_config(config), autoconnect=True)
    if config.midas_configure_hardware:
        hand.configure(enable_torque=False)
    last_command = hand.clip_positions(hand.read_pos())
    hand.set_positions(last_command, clip=True)
    if config.midas_configure_hardware:
        hand.enable_torque()
    print(f"Midas hand connected on {hand.port}; motor_ids={hand.motor_ids}")

    if [int(motor_id) for motor_id in hand.motor_ids] != list(CANONICAL_MIDAS_MOTOR_IDS):
        raise RuntimeError(
            f"Expected Midas motor IDs {CANONICAL_MIDAS_MOTOR_IDS}, got {hand.motor_ids}"
        )

    read_period_ns = int(round(1_000_000_000 / config.midas_measure_rate_hz))
    command_period_ns = int(round(1_000_000_000 / config.midas_command_rate_hz))
    next_read_ns = time.monotonic_ns()
    next_command_ns = next_read_ns
    sample_idx = 0
    sample_session: EpisodeBuffer | None = None
    try:
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()

            if now_ns >= next_command_ns:
                teacher_active, _ = shared.teacher_mode()
                target = shared.latest_midas_target() if teacher_active else None
                if target is not None:
                    target = hand.clip_positions(target * config.midas_command_scale)
                    delta = target - last_command
                    delta *= float(np.clip(config.midas_interpolation_alpha, 0.0, 1.0))
                    if config.midas_max_step_rad > 0:
                        delta = np.clip(
                            delta,
                            -config.midas_max_step_rad,
                            config.midas_max_step_rad,
                        )
                    last_command = hand.clip_positions(last_command + delta)
                    hand.set_positions(last_command, clip=True)
                next_command_ns += command_period_ns
                while next_command_ns < now_ns:
                    next_command_ns += command_period_ns

            if now_ns >= next_read_ns:
                pos, vel, cur = hand.read_pos_vel_cur()
                sample_ns = time.monotonic_ns()
                session = recorder.current()
                if session is not None:
                    if session is not sample_session:
                        sample_session = session
                        sample_idx = 0
                    row: dict[str, object] = {
                        "sample_idx": sample_idx,
                        "monotonic_ns": sample_ns,
                        "wall_time_s": f"{time.time():.6f}",
                        "elapsed_s": f"{session.elapsed_s(sample_ns):.9f}",
                    }
                    for motor_id, value in zip(hand.motor_ids, pos):
                        row[f"pos_{int(motor_id)}"] = f"{float(value):.9f}"
                    for motor_id, value in zip(hand.motor_ids, vel):
                        row[f"vel_{int(motor_id)}"] = f"{float(value):.9f}"
                    for motor_id, value in zip(hand.motor_ids, cur):
                        row[f"cur_{int(motor_id)}"] = f"{float(value):.9f}"
                    session.append_midas(row)
                    sample_idx += 1
                next_read_ns += read_period_ns
                while next_read_ns < sample_ns:
                    next_read_ns += read_period_ns

            next_due_ns = min(next_read_ns, next_command_ns)
            sleep_s = max(0.0, min((next_due_ns - time.monotonic_ns()) / 1e9, 0.005))
            shared.shutdown.wait(sleep_s)
    finally:
        hand.shutdown()
