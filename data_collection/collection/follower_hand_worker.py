from __future__ import annotations

import time

import numpy as np

from .config import CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .hand_common import load_follower_config, verify_connected_hand
from .shared_state import SharedState


def run_follower_hand_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    from midas_hand_api import MidasHand

    if config.follower_hand_config is None or config.follower_hand_port is None:
        raise RuntimeError("Follower worker started without follower configuration")
    hand_config = load_follower_config(config.follower_hand_config, config.follower_hand_port)
    hand = MidasHand(config=hand_config, autoconnect=True)
    verify_connected_hand(hand, role="follower")
    if config.follower_configure_hardware:
        hand.configure(enable_torque=False)

    last_command = hand.clip_positions(hand.read_pos())
    hand.set_positions(last_command, clip=True)
    hand.enable_torque()
    print(
        f"Follower MIDAS connected on {hand.port}; model={hand.config.motor_model}; "
        f"baud={hand.config.baudrate}; motor_ids={hand.motor_ids}"
    )
    init_done.set()

    read_period_ns = int(round(1_000_000_000 / config.follower_measure_rate_hz))
    command_period_ns = int(round(1_000_000_000 / config.follower_command_rate_hz))
    next_read_ns = time.monotonic_ns()
    next_command_ns = next_read_ns
    sample_idx = 0
    sample_session: EpisodeBuffer | None = None
    try:
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()
            if now_ns >= next_command_ns:
                target = shared.latest_hand_target()
                if target is not None:
                    if target.shape != (len(hand.motor_ids),) or not np.isfinite(target).all():
                        raise RuntimeError(
                            f"Primary hand target has shape {target.shape}, but follower "
                            f"expects ({len(hand.motor_ids)},)"
                        )
                    target = hand.clip_positions(target * config.follower_command_scale)
                    delta = target - last_command
                    delta *= config.follower_interpolation_alpha
                    if config.follower_max_step_rad > 0:
                        delta = np.clip(
                            delta,
                            -config.follower_max_step_rad,
                            config.follower_max_step_rad,
                        )
                    last_command = hand.clip_positions(last_command + delta)
                    hand.set_positions(last_command, clip=True)
                next_command_ns += command_period_ns
                while next_command_ns < now_ns:
                    next_command_ns += command_period_ns

            if now_ns >= next_read_ns:
                positions = np.asarray(hand.read_pos(), dtype=np.float64)
                sample_ns = time.monotonic_ns()
                if positions.shape != (len(hand.motor_ids),) or not np.isfinite(positions).all():
                    raise RuntimeError("Follower read returned an invalid motor vector")
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
                    for motor_id, value in zip(hand.motor_ids, positions):
                        row[f"pos_{int(motor_id)}"] = f"{float(value):.9f}"
                    session.append_follower(row)
                    sample_idx += 1
                next_read_ns += read_period_ns
                while next_read_ns < sample_ns:
                    next_read_ns += read_period_ns

            next_due_ns = min(next_read_ns, next_command_ns)
            shared.shutdown.wait(max(0.0, min((next_due_ns - time.monotonic_ns()) / 1e9, 0.005)))
    finally:
        hand.shutdown()
