from __future__ import annotations

import time

import numpy as np

from .config import CollectionConfig
from .episode import EpisodeBuffer, RecordingManager
from .hand_common import load_hand_config, verify_connected_hand
from .shared_state import SharedState


def run_leader_hand_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    from midas_hand_api import MidasHand

    hand_config = load_hand_config(
        config.hand_config, config.hand_port, config.active_motors
    )
    hand = MidasHand(config=hand_config, autoconnect=True)
    verify_connected_hand(hand, role="primary")
    hand.disable_torque()
    print(
        f"TRIMORPH hand connected on {hand.port}; model={hand.config.motor_model}; "
        f"baud={hand.config.baudrate}; motor_ids={hand.motor_ids}"
    )
    init_done.set()

    period_ns = int(round(1_000_000_000 / config.hand_measure_rate_hz))
    next_read_ns = time.monotonic_ns()
    sample_idx = 0
    sample_session: EpisodeBuffer | None = None
    try:
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()
            if now_ns < next_read_ns:
                shared.shutdown.wait(min((next_read_ns - now_ns) / 1e9, 0.005))
                continue

            positions = np.asarray(hand.read_pos(), dtype=np.float64)
            sample_ns = time.monotonic_ns()
            if positions.shape != (len(hand.motor_ids),):
                raise RuntimeError(
                    f"Hand read returned shape {positions.shape}, "
                    f"expected ({len(hand.motor_ids)},)"
                )
            if not np.isfinite(positions).all():
                raise RuntimeError("Hand read returned NaN or Inf")
            shared.set_hand_target(positions)

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
                session.append_leader_hand(row)
                sample_idx += 1

            next_read_ns += period_ns
            while next_read_ns < sample_ns:
                next_read_ns += period_ns
    finally:
        try:
            hand.disable_torque()
        finally:
            hand.shutdown()
