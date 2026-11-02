#!/usr/bin/env python3
"""Quality report for raw collection episodes, run before postprocessing.

    python3 scripts/dataset/check_episodes.py data_collection_0

For every raw/episode_XXXXXX it reports each stream's rate, gaps, and coverage
of the episode, then stream-specific checks: identical hand rows (a still hand
or a failed bus read; the API repeats the last good values for all 13 motors),
camera drops and capture-to-host latency, Quest frozen or jumping positions,
Vicon lost-object samples and occluded markers, and how far each camera frame
is from the nearest hand/Quest/Vicon sample. Episodes without metadata.json
(collection stopped before saving) are listed and skipped. Read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

QUEST_JUMP_M = 0.05
QUEST_FROZEN_SPAN_M = 1e-4


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def stream_line(name: str, stamps: np.ndarray, start_ns: int, stop_ns: int) -> str:
    if len(stamps) < 2:
        return f"  {name:7s} {len(stamps)} samples"
    gaps = np.diff(stamps) / 1e6
    return (f"  {name:7s} n={len(stamps):6d}  {1e9 * (len(stamps) - 1) / (stamps[-1] - stamps[0]):6.1f} Hz  "
            f"gap ms median {np.median(gaps):5.1f} p95 {np.percentile(gaps, 95):5.1f} max {gaps.max():6.1f}  "
            f"covers {(stamps[0] - start_ns) / 1e6:+.0f} .. {(stamps[-1] - stop_ns) / 1e6:+.0f} ms of the episode"
            + ("  NON-INCREASING TIMESTAMPS" if np.any(gaps <= 0) else ""))


def nearest_ms(source: np.ndarray, query: np.ndarray) -> np.ndarray:
    index = np.clip(np.searchsorted(source, query), 1, len(source) - 1)
    return np.minimum(np.abs(query - source[index - 1]), np.abs(source[index] - query)) / 1e6


def longest_run(flags: np.ndarray) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def check_episode(episode: Path, max_sync_error_ms: float) -> list[str]:
    """Print the report for one episode and return its warnings."""

    metadata = json.loads((episode / "metadata.json").read_text())
    start_ns, stop_ns = metadata["started_monotonic_ns"], metadata["stopped_monotonic_ns"]
    warnings: list[str] = []
    print(f"\n=== {episode.name}  status={metadata['status']}  {metadata['duration_s']:.1f} s  "
          f"stop={metadata['stop_reason']}")
    if metadata["status"] != "complete":
        warnings.append(f"status {metadata['status']}: postprocessing skips it")
    for error in metadata.get("errors", []):
        warnings.append(f"{error['worker']} worker error: {error['message']}")

    hand_rows = read_rows(episode / "leader_hand.csv")
    camera_rows = read_rows(episode / "camera_timestamps.csv")
    quest_rows = read_rows(episode / "quest.csv")
    pose_rows = [row for row in read_rows(episode / "mocap.csv") if row["message_type"].endswith("/Position")]
    marker_rows = [row for row in read_rows(episode / "mocap.csv") if row["message_type"].endswith("/MarkerPositions")]
    stamps = {name: np.array([int(row["monotonic_ns"]) for row in rows], dtype=np.int64)
              for name, rows in (("hand", hand_rows), ("camera", camera_rows), ("quest", quest_rows),
                                 ("vicon", pose_rows)) if rows}
    for name, values in stamps.items():
        print(stream_line(name, values, start_ns, stop_ns))
    if "hand" not in stamps:
        warnings.append("no leader-hand samples")
        return warnings

    hand = np.array([[float(row[f"pos_{motor}"]) for motor in range(13)] for row in hand_rows])
    identical = np.all(np.diff(hand, axis=0) == 0, axis=1)
    run = longest_run(identical) + 1
    print(f"  hand    identical consecutive rows {int(identical.sum())}/{len(identical)}, longest run {run} "
          f"({run * 1000 / metadata['trimorph']['measure_rate_hz']:.0f} ms); joint ranges (rad) "
          + " ".join(f"{value:.2f}" for value in np.ptp(hand, axis=0)))
    if run * 1000 / metadata["trimorph"]["measure_rate_hz"] >= 500:
        warnings.append(f"hand identical for {run} samples: still hand or bus dropout (check the terminal log)")
    hand_rate = 1e9 * (len(stamps["hand"]) - 1) / max(stamps["hand"][-1] - stamps["hand"][0], 1)
    if hand_rate < 0.9 * metadata["trimorph"]["measure_rate_hz"]:
        warnings.append(f"hand rate {hand_rate:.1f} Hz below the {metadata['trimorph']['measure_rate_hz']:g} Hz target")

    if camera_rows:
        line = f"  camera  {len(camera_rows)} frames"
        if "sequence_num" in camera_rows[0]:
            sequence = np.array([int(row["sequence_num"]) for row in camera_rows])
            latency = np.array([int(row["receipt_monotonic_ns"]) - int(row["monotonic_ns"]) for row in camera_rows]) / 1e6
            dropped = int(sequence[-1] - sequence[0] + 1 - len(sequence))
            line += f", dropped {dropped}, capture-to-host ms median {np.median(latency):.1f} p95 {np.percentile(latency, 95):.1f}"
            if dropped:
                warnings.append(f"{dropped} camera frames dropped")
        else:
            line += " (receipt timestamps: recorded before capture-time stamping)"
        try:
            import cv2

            capture = cv2.VideoCapture(str(episode / "camera.mp4"))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            line += (f"; video {frames} frames {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                     f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            capture.release()
            if frames != len(camera_rows):
                warnings.append(f"video has {frames} frames but camera_timestamps.csv has {len(camera_rows)}")
        except ImportError:
            pass
        print(line)

    if quest_rows:
        position = np.array([[float(row[key]) for key in ("px", "py", "pz")] for row in quest_rows])
        steps = np.linalg.norm(np.diff(position, axis=0), axis=1)
        sequence = np.array([int(row["seq"]) for row in quest_rows])
        jumps = int(np.sum(steps > QUEST_JUMP_M))
        print(f"  quest   position span {np.ptp(position, axis=0).max():.4f} u, path {steps.sum():.3f} u, "
              f"max step {steps.max(initial=0):.4f} u, jumps > {QUEST_JUMP_M} u: {jumps}, "
              f"seq gaps {int(np.sum(np.diff(sequence) - 1))}")
        if np.ptp(position, axis=0).max() < QUEST_FROZEN_SPAN_M:
            warnings.append("Quest position frozen for the whole episode: positional tracking lost")
        if jumps:
            warnings.append(f"{jumps} Quest position jumps over {QUEST_JUMP_M} u")

    if pose_rows:
        for topic in sorted({row["topic"] for row in pose_rows}):
            rows = [row for row in pose_rows if row["topic"] == topic]
            pose = np.array([[float(row[key]) for key in ("x_trans", "y_trans", "z_trans")] for row in rows])
            lost = np.abs(pose).sum(axis=1) == 0
            frames = np.array([int(row["frame_number"]) for row in rows])
            received = len(np.unique(frames)) / max(frames[-1] - frames[0] + 1, 1)
            valid = pose[~lost]
            path = np.linalg.norm(np.diff(valid, axis=0), axis=1).sum() if len(valid) > 1 else 0.0
            print(f"  vicon   {topic}: lost-object {int(lost.sum())}/{len(rows)} ({100 * lost.mean():.1f}%), "
                  f"Vicon frames received {100 * received:.0f}%, path {path:.0f} mm")
            if lost.mean() > 0.1:
                warnings.append(f"{topic} lost in {100 * lost.mean():.0f}% of samples")
        if marker_rows:
            occluded = sum(row["occluded"] in ("True", "true", "1") for row in marker_rows)
            print(f"  markers {len(marker_rows)} rows, occluded {occluded} ({100 * occluded / len(marker_rows):.1f}%)")

    if "camera" in stamps and len(stamps["camera"]) > 2:
        frames = stamps["camera"][1:-1]  # postprocessing trims one frame at each end
        sources = {"hand": stamps["hand"]}
        if "quest" in stamps:
            sources["quest"] = stamps["quest"]
        if pose_rows:
            sources["vicon"] = np.sort(np.array([int(row["monotonic_ns"]) for row in pose_rows
                                                 if any(float(row[k]) for k in ("x_trans", "y_trans", "z_trans"))]))
        for name, source in sources.items():
            if len(source) < 2:
                continue
            distance = nearest_ms(source, frames)
            late = int(np.sum(distance > max_sync_error_ms))
            print(f"  sync    camera->{name:5s} nearest ms median {np.median(distance):5.1f} "
                  f"p95 {np.percentile(distance, 95):6.1f} max {distance.max():7.1f}; over {max_sync_error_ms:g} ms: "
                  f"{late}/{len(distance)}")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--max-sync-error-ms", type=float, default=10.0)
    args = parser.parse_args()
    episodes = sorted(path for path in (args.dataset_root / "raw").glob("episode_*") if path.is_dir())
    if not episodes:
        raise SystemExit(f"No episodes under {args.dataset_root / 'raw'}")
    summary = {}
    for episode in episodes:
        if not (episode / "metadata.json").exists():
            summary[episode.name] = ["no metadata.json: collection stopped before saving; delete it before postprocessing"]
            print(f"\n=== {episode.name}  NOT SAVED (files: {', '.join(sorted(p.name for p in episode.iterdir()))})")
            continue
        summary[episode.name] = check_episode(episode, args.max_sync_error_ms)
    print("\n=== Summary")
    for name, warnings in summary.items():
        print(f"  {name}: " + ("OK" if not warnings else "; ".join(warnings)))
    return 1 if any(summary.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
