"""Wire protocol for the cuRobo MPC client (robot box) <-> solver (GPU box) link.

Reuses the framed-TCP wire format already proven in inference/protocol.py
(predictor <-> executor): [u32 header_len][header json][blobs...]. The control
pattern here is the same synchronous request/reply shape (curobo_worker.py's
25Hz loop stalls on each plan() call either way, whether solved locally or
remotely), so no new transport is introduced -- just cuRobo-specific framing on
top of the existing send_msg/recv_msg helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INFERENCE_DIR = _REPO_ROOT / "inference"
if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))

import protocol  # noqa: E402

set_low_latency = protocol.set_low_latency


def send_plan_request(
    sock,
    q_current: list[float],
    target_eef: list[float],
    reset: bool,
) -> None:
    header = {"reset": bool(reset)}
    q_blob = np.asarray(q_current, dtype=np.float64).tobytes()
    eef_blob = np.asarray(target_eef, dtype=np.float64).tobytes()
    protocol.send_msg(sock, header, q_blob, eef_blob)


def recv_plan_request(
    sock,
) -> Optional[Tuple[list[float], list[float], bool]]:
    msg = protocol.recv_msg(sock)
    if msg is None:
        return None
    header, blobs = msg
    q_current = np.frombuffer(blobs[0], dtype=np.float64).tolist()
    target_eef = np.frombuffer(blobs[1], dtype=np.float64).tolist()
    return q_current, target_eef, bool(header["reset"])


def send_plan_response(
    sock,
    pos: Optional[np.ndarray],
    vel: Optional[np.ndarray],
) -> None:
    if pos is None:
        protocol.send_msg(sock, {"ok": False})
        return
    header = {"ok": True, "n_waypoints": int(pos.shape[0])}
    pos_blob = np.ascontiguousarray(pos, dtype=np.float64).tobytes()
    vel_blob = np.ascontiguousarray(vel, dtype=np.float64).tobytes()
    protocol.send_msg(sock, header, pos_blob, vel_blob)


def recv_plan_response(sock) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    msg = protocol.recv_msg(sock)
    if msg is None:
        return None
    header, blobs = msg
    if not header.get("ok", False):
        return None
    n = int(header["n_waypoints"])
    pos = np.frombuffer(blobs[0], dtype=np.float64).reshape(n, 6).copy()
    vel = np.frombuffer(blobs[1], dtype=np.float64).reshape(n, 6).copy()
    return pos, vel
