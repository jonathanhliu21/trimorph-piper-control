from __future__ import annotations

import time

from .config import TACTILE_FINGER_ORDER, CollectionConfig
from .episode import RecordingManager
from .shared_state import SharedState


def run_tactile_worker(
    config: CollectionConfig,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    if not config.enable_tactile:
        init_done.set()
        return
    try:
        from midas_hand_api import PaxiniConfig, PaxiniHandSensor

        tactile_config = PaxiniConfig(
            port=config.tactile_port,
            fingers=list(config.tactile_fingers),
            publish_rate_hz=config.tactile_publish_rate_hz,
        )
        sensor = PaxiniHandSensor(tactile_config)
        connect_exc: Exception | None = None
        for attempt in range(3):
            try:
                sensor.connect()
                connect_exc = None
                break
            except Exception as exc:
                connect_exc = exc
                if attempt < 2:
                    print(f"Tactile connect attempt {attempt + 1} failed, retrying in 2s: {exc}")
                    time.sleep(2.0)
        if connect_exc is not None:
            raise connect_exc
    except Exception:
        if config.require_tactile:
            raise
        print("Warning: tactile sensor unavailable; continuing without tactile data.")
        init_done.set()
        return

    print(f"Tactile sensor connected on {sensor.port}; fingers={config.tactile_fingers}")
    init_done.set()
    period_ns = int(round(1_000_000_000 / config.tactile_rate_hz))
    next_sample_ns = time.monotonic_ns()
    received_first_frame = False
    try:
        while not shared.shutdown.is_set():
            now_ns = time.monotonic_ns()
            if now_ns < next_sample_ns:
                shared.shutdown.wait(min((next_sample_ns - now_ns) / 1e9, 0.01))
                continue
            try:
                data = sensor.read_latest()
            except RuntimeError as exc:
                if not received_first_frame and "No tactile data available yet" in str(exc):
                    next_sample_ns += period_ns
                    continue
                if not config.require_tactile:
                    print(f"Warning: tactile sensor stopped: {exc}")
                    return
                raise RuntimeError(f"Tactile sensor stopped during collection: {exc}") from exc

            received_first_frame = True
            missing_fingers = [
                finger for finger in config.tactile_fingers if finger not in data
            ]
            if missing_fingers:
                if not config.require_tactile:
                    print(f"Warning: tactile frame missing fingers: {missing_fingers}")
                    return
                raise RuntimeError(f"Tactile frame missing fingers: {missing_fingers}")

            session = recorder.current()
            if session is None:
                next_sample_ns += period_ns
                continue

            sample_ns = time.monotonic_ns()
            session.append_tactile(sample_ns, time.time(), data)
            next_sample_ns += period_ns
            while next_sample_ns < sample_ns:
                next_sample_ns += period_ns
    finally:
        sensor.disconnect()
