"""MIDAS inference predictor.

Runs on the GPU box. Loads the diffusion policy, then serves a synchronous
request/reply loop over TCP: receive (image, robot_state) from the executor,
run inference, send back an absolute action chunk of shape (T, 19).

The policy wrapper converts its relative predictions to absolute commands using
the robot_state we are sent, so the executor can apply the chunk directly.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

INFERENCE_DIR = Path(__file__).resolve().parents[1]
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

import protocol  # noqa: E402

DEFAULT_POLICY_REPO = "/home/alvinz/alvinz/diffusion-policies"
DEFAULT_CHECKPOINT = (
    "/home/alvinz/alvinz/diffusion-policies/data_midas1/outputs/2026.05.23/"
    "19.27.53_train_diffusion_midas_image_hand_cond_dinov2_16_midas_image_hand_cond/"
    "checkpoints/latest.ckpt"
)


def parse_args():
    p = argparse.ArgumentParser(description="MIDAS inference predictor (server).")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--policy-repo", default=DEFAULT_POLICY_REPO,
                   help="Path to the diffusion-policies repo (for imports).")
    p.add_argument("--host", default="0.0.0.0", help="Bind address.")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--device", default=None, help="cuda / cuda:0 / cpu. Default: auto.")
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="Override sampler steps (lower = faster, e.g. DDIM 8-16).")
    p.add_argument("--action-mode", default="chunk", choices=["first", "chunk", "all"])
    p.add_argument("--no-ema", action="store_true", help="Use the raw (non-EMA) policy.")
    p.add_argument("--warmup", type=int, default=3, help="Warmup inferences at startup.")
    return p.parse_args()


def load_policy(args):
    if args.policy_repo and args.policy_repo not in sys.path:
        sys.path.insert(0, args.policy_repo)
    from shared.env.midas.midas_inference import MidasPolicyInference

    print(f"Loading checkpoint:\n  {args.checkpoint}")
    policy = MidasPolicyInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_ema=not args.no_ema,
        num_inference_steps=args.num_inference_steps,
        action_mode=args.action_mode,
    )
    print(f"Policy ready on {policy.device}: "
          f"image={policy.expected_chw}, action_dim={policy.action_dim}, "
          f"obs_steps={policy.num_obs_steps}, steps={policy.policy.num_inference_steps}")
    return policy


def warmup(policy, n: int) -> None:
    if n <= 0:
        return
    h, w = int(policy.expected_chw[1]), int(policy.expected_chw[2])
    dummy_img = np.zeros((h, w, 3), dtype=np.uint8)
    dummy_state = np.zeros(policy.action_dim, dtype=np.float32)
    print(f"Warming up ({n}x)...")
    for _ in range(n):
        policy.reset()
        policy.predict(dummy_img, dummy_state)
    policy.reset()
    print("Warmup done.")


def serve_connection(policy, conn: socket.socket, peer) -> None:
    protocol.set_low_latency(conn)
    policy.reset()
    print(f"Executor connected: {peer}")
    count = 0
    lat_sum = 0.0
    try:
        while True:
            obs = protocol.recv_obs(conn)
            if obs is None:
                print(f"Executor {peer} disconnected.")
                return
            seq, image_rgb, state, _ = obs

            t0 = time.perf_counter()
            result = policy.predict(image_rgb, state)
            infer_ms = (time.perf_counter() - t0) * 1e3

            protocol.send_action(conn, seq, result["action"], infer_ms=infer_ms)

            count += 1
            lat_sum += infer_ms
            if count % 20 == 0:
                print(f"  seq={seq} infer={infer_ms:6.1f}ms "
                      f"(avg {lat_sum / count:6.1f}ms)")
    except (ConnectionResetError, BrokenPipeError):
        print(f"Executor {peer} connection dropped.")


def main():
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    policy = load_policy(args)
    warmup(policy, args.warmup)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"Listening on {args.host}:{args.port}  (Ctrl-C to stop)")

    try:
        while True:
            conn, peer = server.accept()
            with conn:
                serve_connection(policy, conn, peer)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
