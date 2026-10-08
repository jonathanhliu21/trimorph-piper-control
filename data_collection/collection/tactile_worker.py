"""Paxini tactile sensor worker for hand-only collection."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .shared_state import SharedState
from .tactile_contact_detection import TactileContactDetector


TACTILE_SENSOR_NAMES = ("index", "middle", "ring", "thumb")
TACTILE_CONTACT_LOGGER_NAME = TactileContactDetector.__module__
TACTILE_POLL_RATE_HZ = 60.0


def configure_tactile_debug_log(path: Path) -> logging.Handler:
    """Attach the structured detector log used by the standalone hand UI."""

    log_path = path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%H:%M:%S"
        )
    )
    tactile_logger = logging.getLogger(TACTILE_CONTACT_LOGGER_NAME)
    tactile_logger.setLevel(logging.DEBUG)
    tactile_logger.addHandler(handler)
    tactile_logger.propagate = False
    tactile_logger.info("TACTILE_CONTACT event=log_started path=%s", log_path)
    return handler


def run_tactile_worker(config, shared: SharedState, _recorder, *, init_done) -> None:
    """Read Paxini normal forces and publish four binary contact states."""

    from midas_hand_api import PaxiniConfig, PaxiniHandSensor

    debug_handler = None
    debug_path = getattr(config, "tactile_debug_log", None)
    if debug_path is not None:
        debug_handler = configure_tactile_debug_log(debug_path)

    sensor = PaxiniHandSensor(PaxiniConfig(port=config.paxini_port))
    detectors = {
        name: TactileContactDetector(name=name) for name in TACTILE_SENSOR_NAMES
    }
    try:
        sensor.connect()
        shared.set_tactile_contacts((False,) * len(TACTILE_SENSOR_NAMES))
        print(f"Paxini tactile sensor connected on {sensor.port}")
        init_done.set()

        period_s = 1.0 / TACTILE_POLL_RATE_HZ
        next_tick_s = time.monotonic()
        while not shared.shutdown.is_set():
            now_s = time.monotonic()
            try:
                tactile_frame = sensor.read_tactile_fz()
            except RuntimeError as exc:
                if "No tactile data available yet" not in str(exc):
                    raise
            else:
                missing = [
                    name for name in TACTILE_SENSOR_NAMES if name not in tactile_frame
                ]
                if missing:
                    raise RuntimeError(f"Tactile frame is missing sensors: {missing}")
                shared.set_tactile_contacts(
                    tuple(
                        detectors[name].update(tactile_frame[name], now_s=now_s)
                        for name in TACTILE_SENSOR_NAMES
                    )
                )

            next_tick_s += period_s
            wait_s = next_tick_s - time.monotonic()
            if wait_s <= 0:
                next_tick_s = time.monotonic()
                continue
            shared.shutdown.wait(wait_s)
    finally:
        shared.set_tactile_contacts(None)
        sensor.disconnect()
        if debug_handler is not None:
            tactile_logger = logging.getLogger(TACTILE_CONTACT_LOGGER_NAME)
            tactile_logger.removeHandler(debug_handler)
            debug_handler.close()
