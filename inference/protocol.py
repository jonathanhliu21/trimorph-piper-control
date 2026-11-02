"""Wire protocol for the predictor <-> executor link.

Deliberately dependency-free (stdlib ``socket`` + numpy + cv2). The control
pattern is strictly synchronous request/reply (the executor stalls while the
predictor runs), so a plain framed TCP socket is all we need -- no ZeroMQ.

Frame layout (big-endian):

    [u32 header_len][header json][blob 0][blob 1]...

The JSON header carries metadata plus ``blob_sizes`` so the receiver knows how
many bytes to read for each binary blob. Observations send a JPEG image blob +
a raw float32 state blob; action replies send a single raw float32 blob.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ACTION_DIM = 19  # [eef_xyz_rotvec (6), hand motors (13)]


# --------------------------------------------------------------------------- #
# Socket helpers
# --------------------------------------------------------------------------- #
def set_low_latency(sock: socket.socket) -> None:
    """Disable Nagle's algorithm; tiny messages must not be buffered/coalesced."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly ``n`` bytes, or return None on clean EOF."""
    chunks = bytearray()
    while len(chunks) < n:
        chunk = sock.recv(n - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def send_msg(sock: socket.socket, header: Dict, *blobs: bytes) -> None:
    header = dict(header)
    header["blob_sizes"] = [len(b) for b in blobs]
    hb = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack(">I", len(hb)))
    sock.sendall(hb)
    for b in blobs:
        sock.sendall(b)


def recv_msg(sock: socket.socket) -> Optional[Tuple[Dict, List[bytes]]]:
    """Receive one framed message, or None if the peer closed the connection."""
    raw = _recvall(sock, 4)
    if raw is None:
        return None
    (hlen,) = struct.unpack(">I", raw)
    hb = _recvall(sock, hlen)
    if hb is None:
        return None
    header = json.loads(hb.decode("utf-8"))
    blobs: List[bytes] = []
    for n in header.get("blob_sizes", []):
        blob = _recvall(sock, n)
        if blob is None:
            return None
        blobs.append(blob)
    return header, blobs


# --------------------------------------------------------------------------- #
# Image / array codecs
# --------------------------------------------------------------------------- #
def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """JPEG-encode an HWC uint8 image. Channel order round-trips, so feeding RGB
    in yields RGB out on the other end (cv2's BGR assumption is symmetric)."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def decode_jpeg(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cv2.imdecode failed")
    return image


# --------------------------------------------------------------------------- #
# High-level obs / action messages
# --------------------------------------------------------------------------- #
def send_obs(
    sock: socket.socket,
    seq: int,
    image_rgb: np.ndarray,
    state: np.ndarray,
    *,
    t_capture: float,
    jpeg_quality: int = 85,
) -> None:
    state = np.ascontiguousarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise ValueError("Observation state must be finite [19]")
    header = {
        "type": "obs",
        "seq": int(seq),
        "t_capture": float(t_capture),
        "img_shape": list(image_rgb.shape),
        "state_dim": int(state.shape[0]),
    }
    send_msg(sock, header, encode_jpeg(image_rgb, jpeg_quality), state.tobytes())


def recv_obs(
    sock: socket.socket,
) -> Optional[Tuple[int, np.ndarray, np.ndarray, Dict]]:
    msg = recv_msg(sock)
    if msg is None:
        return None
    header, blobs = msg
    if header.get("type") != "obs" or len(blobs) != 2:
        raise ValueError("Expected an observation message")
    image_rgb = decode_jpeg(blobs[0])
    state = np.frombuffer(blobs[1], dtype=np.float32).copy()
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise ValueError("Observation state must be finite [19]")
    return int(header["seq"]), image_rgb, state, header


def send_action(
    sock: socket.socket,
    seq: int,
    action_chunk: np.ndarray,
    *,
    infer_ms: float,
) -> None:
    chunk = np.ascontiguousarray(np.atleast_2d(action_chunk), dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or not len(chunk) or not np.isfinite(chunk).all():
        raise ValueError("Action chunk must be finite, nonempty [T,19]")
    header = {
        "type": "action",
        "seq": int(seq),
        "shape": list(chunk.shape),
        "infer_ms": float(infer_ms),
    }
    send_msg(sock, header, chunk.tobytes())


def recv_action(
    sock: socket.socket,
) -> Optional[Tuple[int, np.ndarray, Dict]]:
    msg = recv_msg(sock)
    if msg is None:
        return None
    header, blobs = msg
    if header.get("type") != "action" or len(blobs) != 1:
        raise ValueError("Expected an action message")
    shape = tuple(header["shape"])
    chunk = np.frombuffer(blobs[0], dtype=np.float32).reshape(shape).copy()
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or not len(chunk) or not np.isfinite(chunk).all():
        raise ValueError("Action chunk must be finite, nonempty [T,19]")
    return int(header["seq"]), chunk, header
