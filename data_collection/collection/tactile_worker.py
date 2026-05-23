from __future__ import annotations

import time

from .config import TACTILE_FINGER_ORDER, CollectionConfig
from .episode import RecordingManager
from .shared_state import SharedState


def run_tactile_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
) -> None:
    if not config.enable_tactile:
        return
    try:
        from midas_hand_api import PaxiniConfig, PaxiniHandSensor

        tactile_config = PaxiniConfig(
            port=config.tactile_port,
            fingers=list(config.tactile_fingers),
            publish_rate_hz=config.tactile_publish_rate_hz,
        )
        sensor = PaxiniHandSensor(tactile_config)
        sensor.connect()
    except Exception:
        if config.require_tactile:
            raise
        print("Warning: tactile sensor unavailable; continuing without tactile data.")
        return

    print(f"Tactile sensor connected on {sensor.port}; fingers={config.tactile_fingers}")
    period_ns = int(round(1_000_000_000 / config.tactile_rate_hz))
    next_sample_ns = time.monotonic_ns()
    last_signature: tuple[int, ...] | None = None
    try:
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()
            if now_ns < next_sample_ns:
                shared.shutdown.wait(min((next_sample_ns - now_ns) / 1e9, 0.01))
                continue
            try:
                data = sensor.read_latest()
            except RuntimeError:
                next_sample_ns += period_ns
                continue

            sample_ns = time.monotonic_ns()
            signature = tuple(id(data.get(finger)) for finger in TACTILE_FINGER_ORDER)
            if signature != last_signature:
                session = recorder.current()
                if session is not None:
                    session.append_tactile(sample_ns, time.time(), data)
                last_signature = signature
            next_sample_ns += period_ns
            while next_sample_ns < sample_ns:
                next_sample_ns += period_ns
    finally:
        sensor.disconnect()
