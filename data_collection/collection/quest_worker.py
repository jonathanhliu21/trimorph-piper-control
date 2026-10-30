from __future__ import annotations

import json
import math
import socket
import time
from typing import Any

from .config import QUEST_PACKET_FIELDS
from .episode import EpisodeBuffer, RecordingManager
from .shared_state import SharedState


QUEST_POSE_FIELDS = QUEST_PACKET_FIELDS[1:]
QUEST_MAX_DATAGRAM_BYTES = 2048
QUEST_SOCKET_POLL_TIMEOUT_S = 0.2


def _decode_packet(data: bytes) -> dict[str, int | float]:
    try:
        packet: Any = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Quest UDP JSON packet: {exc}") from exc

    if not isinstance(packet, dict):
        raise ValueError("Invalid Quest UDP JSON packet: expected an object")

    missing = [field for field in QUEST_PACKET_FIELDS if field not in packet]
    if missing:
        raise ValueError(
            "Invalid Quest UDP JSON packet: missing fields " + ", ".join(missing)
        )

    seq = packet["seq"]
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise ValueError("Invalid Quest UDP JSON packet: seq must be an integer")

    result: dict[str, int | float] = {"seq": seq}
    for field in QUEST_POSE_FIELDS:
        value = packet[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"Invalid Quest UDP JSON packet: {field} must be a finite number"
            )
        result[field] = value
    return result


def run_quest_worker(
    config,
    shared: SharedState,
    recorder: RecordingManager,
    *,
    init_done,
) -> None:
    """Receive raw Quest poses over UDP and record them with host timestamps."""

    # Collection enables this; standalone tools can keep their own jump reporting.
    jump_threshold = getattr(config, "quest_jump_threshold", None)
    if jump_threshold is not None and (not math.isfinite(jump_threshold) or jump_threshold <= 0):
        raise ValueError("quest_jump_threshold must be finite and positive")
    previous_position = None
    previous_sample_ns = None
    first_sample_ns = None
    jump_count = 0
    startup_deadline = time.monotonic() + config.quest_startup_timeout_s
    initialized = False
    sample_idx = 0
    sample_session: EpisodeBuffer | None = None

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((config.quest_host, config.quest_port))
        print(
            f"[quest_worker] Listening on UDP {config.quest_host}:"
            f"{config.quest_port}"
        )

        while not shared.shutdown.is_set():
            if initialized:
                timeout_s = QUEST_SOCKET_POLL_TIMEOUT_S
            else:
                remaining_s = startup_deadline - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError(
                        "No valid Quest UDP data received within "
                        f"{config.quest_startup_timeout_s:g}s"
                    )
                timeout_s = min(QUEST_SOCKET_POLL_TIMEOUT_S, remaining_s)
            sock.settimeout(timeout_s)

            try:
                data, _address = sock.recvfrom(QUEST_MAX_DATAGRAM_BYTES)
            except socket.timeout:
                continue

            sample_ns = time.monotonic_ns()
            wall_time_s = time.time()
            packet = _decode_packet(data)
            if jump_threshold is not None:
                position = tuple(float(packet[key]) for key in ("px", "py", "pz"))
                if first_sample_ns is None:
                    first_sample_ns = sample_ns
                if previous_position is not None:
                    delta = tuple(value - previous for value, previous in zip(position, previous_position))
                    distance = math.hypot(*delta)
                    if distance > jump_threshold:
                        jump_count += 1
                        print(
                            f"WARNING: [quest_worker] POSITION JUMP #{jump_count} "
                            f"t={(sample_ns - first_sample_ns) / 1e9:.3f}s "
                            f"seq={packet['seq']} step={distance:.4f} units "
                            f"dt={(sample_ns - previous_sample_ns) / 1e9:.4f}s "
                            f"delta=({delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}) "
                            f"threshold={jump_threshold:g}",
                            flush=True,
                        )
                previous_position = position
                previous_sample_ns = sample_ns

            if not initialized:
                initialized = True
                init_done.set()

            session = recorder.current()
            if session is None:
                continue
            if session is not sample_session:
                sample_session = session
                sample_idx = 0

            row: dict[str, object] = {
                "sample_idx": sample_idx,
                "monotonic_ns": sample_ns,
                "wall_time_s": f"{wall_time_s:.6f}",
                "elapsed_s": f"{session.elapsed_s(sample_ns):.9f}",
            }
            row.update(packet)
            session.append_quest(row)
            sample_idx += 1

        if not initialized:
            raise RuntimeError(
                "Quest worker stopped before receiving a valid UDP packet"
            )
