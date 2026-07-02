#!/usr/bin/env python3
"""cuRobo MPC solver server: runs on a second GPU box, serves plan() over TCP.

Offloads data_collection/collection/curobo_worker.py's MPC solve to a separate
machine (see CUROBO_REMOTE_ENABLED in data_collection/collection/config.py).
Mirrors inference/predictor/predictor.py's synchronous request/reply shape,
reusing the same framed-TCP wire protocol
(scripts/piper_api/curobo_remote_protocol.py -> inference/protocol.py).

Usage:
    python3 scripts/curobo_solver_server.py --host 0.0.0.0 --port 5556

This box needs its own checkout of this repo with curobo + a CUDA-enabled
torch installed, and assets/models/piper_arm.urdf/.xrdf present at the same
relative path. Its own config.py should leave CUROBO_REMOTE_ENABLED=False
(only the robot PC's config enables it) -- this script refuses to start
otherwise, to avoid a solver box trying to call itself remotely.

Deliberately does NOT call data_collection.collection.config.validate_config():
that function also checks hardware this box doesn't have (task camera,
RealSense, CAN interface, delta-teleop config path, ...), which don't apply
here and would spuriously fail startup.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_COLLECTION_DIR = REPO_ROOT / "data_collection"
for _p in (REPO_ROOT, DATA_COLLECTION_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from collection.config import build_config  # noqa: E402
from collection.curobo_worker import _create_curobo_planner  # noqa: E402
from scripts.piper_api import curobo_remote_protocol as protocol  # noqa: E402

_HOME_EEF = [0.325, -0.3, 0.2, -1.5707963267948966, -0.659612, -1.5707963267948966]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cuRobo MPC solver server (GPU box).")
    p.add_argument("--host", default="0.0.0.0", help="Bind address.")
    p.add_argument("--port", type=int, default=5556)
    return p.parse_args()


def warmup(plan) -> None:
    print("Warming up MPC solver (cold-start solve)...")
    plan([0.0] * 6, _HOME_EEF, reset=False)
    print("Warmup done.")


def serve_connection(plan, conn: socket.socket, peer) -> None:
    protocol.set_low_latency(conn)
    print(f"Client connected: {peer}")
    count = 0
    try:
        while True:
            req = protocol.recv_plan_request(conn)
            if req is None:
                print(f"Client {peer} disconnected.")
                return
            q_current, target_eef, reset = req
            try:
                planned = plan(q_current, target_eef, reset=reset)
            except Exception as exc:  # keep serving; report failure to this client
                print(f"[curobo_solver_server] plan error: {exc}")
                planned = None
            if planned is None:
                protocol.send_plan_response(conn, None, None)
            else:
                pos, vel = planned
                protocol.send_plan_response(conn, pos, vel)
            count += 1
            if count % 100 == 0:
                print(f"  served {count} requests from {peer}")
    except (ConnectionResetError, BrokenPipeError):
        print(f"Client {peer} connection dropped.")


def main() -> int:
    args = parse_args()

    config = build_config()
    if config.curobo_remote_enabled:
        print(
            "This box's own config.py has CUROBO_REMOTE_ENABLED=True, which "
            "would make the solver try to call itself remotely. Set it to "
            "False on this machine -- only the robot PC's config should "
            "enable it.",
            file=sys.stderr,
        )
        return 1

    plan, _is_continuous = _create_curobo_planner(config)
    warmup(plan)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"Listening on {args.host}:{args.port}  (Ctrl-C to stop)")

    try:
        while True:
            conn, peer = server.accept()
            with conn:
                serve_connection(plan, conn, peer)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
