#!/usr/bin/env python3
"""Convert raw Midas/Piper collection episodes into diffusion-policy files."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# Operator settings. Adjust these values, then run:
#   python3 data_collection/postprocess_dataset.py
DATASET_ROOT = Path("data_midas")
RAW_DIR = DATASET_ROOT / "raw"
OVERWRITE_PROCESSED_FILES = True
PROCESS_ONLY_RAW_STATUSES = {"complete"}
MAX_SYNC_ERROR_S = 0.010
DROP_FRAMES_OUTSIDE_MAX_SYNC_ERROR = False


CANONICAL_MIDAS_MOTOR_IDS = tuple(range(13))
RAW_EEF_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class PostprocessConfig:
    dataset_root: Path
    raw_dir: Path
    overwrite_processed_files: bool
    process_only_raw_statuses: set[str]
    max_sync_error_s: float
    drop_frames_outside_max_sync_error: bool


@dataclass(frozen=True)
class RawEpisode:
    name: str
    index: int
    path: Path
    camera_path: Path
    camera_timestamps_path: Path
    midas_path: Path
    piper_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class ProcessedPaths:
    video_path: Path
    hand_path: Path
    eef_path: Path
    sync_path: Path


def build_config() -> PostprocessConfig:
    dataset_root = Path(DATASET_ROOT).expanduser()
    raw_dir = Path(RAW_DIR).expanduser()
    return PostprocessConfig(
        dataset_root=dataset_root,
        raw_dir=raw_dir,
        overwrite_processed_files=OVERWRITE_PROCESSED_FILES,
        process_only_raw_statuses=set(PROCESS_ONLY_RAW_STATUSES),
        max_sync_error_s=MAX_SYNC_ERROR_S,
        drop_frames_outside_max_sync_error=DROP_FRAMES_OUTSIDE_MAX_SYNC_ERROR,
    )


def validate_config(config: PostprocessConfig) -> None:
    if config.max_sync_error_s <= 0:
        raise ValueError("MAX_SYNC_ERROR_S must be greater than 0")
    if not config.raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {config.raw_dir}")


def processed_paths(dataset_root: Path, episode_index: int) -> ProcessedPaths:
    stem = f"{episode_index:06d}"
    return ProcessedPaths(
        video_path=dataset_root / "camera" / "mp4_files" / f"{stem}.mp4",
        hand_path=dataset_root / "processed_encoder_t" / f"{stem}.npy",
        eef_path=dataset_root / "absolute_action_t" / f"{stem}.npy",
        sync_path=dataset_root / "syncs" / f"{stem}.npz",
    )


def ensure_processed_dirs(dataset_root: Path) -> None:
    for rel_path in (
        "camera/mp4_files",
        "processed_encoder_t",
        "absolute_action_t",
        "syncs",
    ):
        (dataset_root / rel_path).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_raw_episodes(config: PostprocessConfig) -> list[RawEpisode]:
    episodes = []
    for episode_dir in sorted(config.raw_dir.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        suffix = episode_dir.name.removeprefix("episode_")
        if not suffix.isdigit():
            continue
        episode = RawEpisode(
            name=episode_dir.name,
            index=int(suffix),
            path=episode_dir,
            camera_path=episode_dir / "camera.mp4",
            camera_timestamps_path=episode_dir / "camera_timestamps.csv",
            midas_path=episode_dir / "midas.csv",
            piper_path=episode_dir / "piper.csv",
            metadata_path=episode_dir / "metadata.json",
        )
        episodes.append(episode)
    return episodes


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def int_column(rows: list[dict[str, str]], field: str, path: Path) -> np.ndarray:
    try:
        return np.asarray([int(row[field]) for row in rows], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Missing column {field!r} in {path}") from exc


def require_monotonic_ns(name: str, timestamps_ns: np.ndarray, path: Path) -> None:
    if timestamps_ns.ndim != 1 or timestamps_ns.size == 0:
        raise ValueError(f"{name} timestamps are empty in {path}")
    if np.any(np.diff(timestamps_ns) < 0):
        raise ValueError(f"{name} timestamps are not monotonic in {path}")


def load_camera_timestamps(episode: RawEpisode) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(episode.camera_timestamps_path)
    if not rows:
        raise ValueError(f"No camera timestamps in {episode.camera_timestamps_path}")
    frame_idx = int_column(rows, "frame_idx", episode.camera_timestamps_path)
    timestamps_ns = int_column(rows, "monotonic_ns", episode.camera_timestamps_path)
    require_monotonic_ns("camera", timestamps_ns, episode.camera_timestamps_path)
    return frame_idx, timestamps_ns


def load_midas(episode: RawEpisode) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(episode.midas_path)
    if not rows:
        raise ValueError(f"No Midas samples in {episode.midas_path}")
    timestamps_ns = int_column(rows, "monotonic_ns", episode.midas_path)
    require_monotonic_ns("Midas", timestamps_ns, episode.midas_path)

    positions = []
    for row in rows:
        positions.append([float(row[f"pos_{motor_id}"]) for motor_id in CANONICAL_MIDAS_MOTOR_IDS])
    hand = np.asarray(positions, dtype=np.float32)
    if hand.ndim != 2 or hand.shape[1] != 13:
        raise ValueError(f"Midas array has invalid shape {hand.shape} for {episode.name}")
    if not np.isfinite(hand).all():
        raise ValueError(f"Midas array contains NaN or Inf for {episode.name}")
    return hand, timestamps_ns


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cos_angle = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    angle = math.acos(cos_angle)
    if angle < 1e-12:
        return 0.5 * np.asarray(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        )
    if math.pi - angle < 1e-6:
        axis = np.sqrt(
            np.maximum(
                0.0,
                np.asarray(
                    [
                        (rotation[0, 0] + 1.0) * 0.5,
                        (rotation[1, 1] + 1.0) * 0.5,
                        (rotation[2, 2] + 1.0) * 0.5,
                    ],
                    dtype=np.float64,
                ),
            )
        )
        axis[0] = math.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = math.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        axis[2] = math.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = axis / norm
        return axis * angle
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * math.sin(angle)
    return axis * angle


def euler_xyz_to_rotvec(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return matrix_to_rotvec(euler_xyz_to_matrix(roll, pitch, yaw))


def load_piper(episode: RawEpisode) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(episode.piper_path)
    if not rows:
        raise ValueError(f"No Piper samples in {episode.piper_path}")
    timestamps_ns = int_column(rows, "monotonic_ns", episode.piper_path)
    require_monotonic_ns("Piper", timestamps_ns, episode.piper_path)

    poses = []
    for row in rows:
        x, y, z, roll, pitch, yaw = [float(row[field]) for field in RAW_EEF_FIELDS]
        rotvec = euler_xyz_to_rotvec(roll, pitch, yaw)
        poses.append([x, y, z, float(rotvec[0]), float(rotvec[1]), float(rotvec[2])])
    eef = np.asarray(poses, dtype=np.float32)
    if eef.ndim != 2 or eef.shape[1] != 6:
        raise ValueError(f"Piper EEF array has invalid shape {eef.shape} for {episode.name}")
    if not np.isfinite(eef).all():
        raise ValueError(f"Piper EEF array contains NaN or Inf for {episode.name}")
    return eef, timestamps_ns


def nearest_indices(source_ns: np.ndarray, query_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if source_ns.size == 0:
        raise ValueError("Cannot sync against an empty timestamp array")
    insertion = np.searchsorted(source_ns, query_ns)
    left = np.clip(insertion - 1, 0, source_ns.size - 1)
    right = np.clip(insertion, 0, source_ns.size - 1)
    left_delta = np.abs(query_ns - source_ns[left])
    right_delta = np.abs(source_ns[right] - query_ns)
    use_right = right_delta < left_delta
    indices = np.where(use_right, right, left).astype(np.int64)
    deltas_ns = np.abs(source_ns[indices] - query_ns)
    return indices, deltas_ns


def make_sync(
    config: PostprocessConfig,
    frame_idx: np.ndarray,
    camera_ns: np.ndarray,
    hand_ns: np.ndarray,
    eef_ns: np.ndarray,
    episode: RawEpisode,
) -> dict[str, np.ndarray]:
    encoder_idx, encoder_delta_ns = nearest_indices(hand_ns, camera_ns)
    pose_idx, pose_delta_ns = nearest_indices(eef_ns, camera_ns)
    max_delta_ns = int(round(config.max_sync_error_s * 1_000_000_000))
    valid = (encoder_delta_ns <= max_delta_ns) & (pose_delta_ns <= max_delta_ns)

    if not np.all(valid):
        bad = int(np.count_nonzero(~valid))
        worst_hand_s = float(encoder_delta_ns.max() / 1e9)
        worst_eef_s = float(pose_delta_ns.max() / 1e9)
        message = (
            f"{episode.name} has {bad} camera frames outside MAX_SYNC_ERROR_S="
            f"{config.max_sync_error_s:.3f}; worst hand={worst_hand_s:.6f}s, "
            f"worst eef={worst_eef_s:.6f}s"
        )
        if not config.drop_frames_outside_max_sync_error:
            raise ValueError(message)
        print(f"Warning: {message}; dropping those sync rows.")
        frame_idx = frame_idx[valid]
        encoder_idx = encoder_idx[valid]
        pose_idx = pose_idx[valid]

    if frame_idx.size == 0:
        raise ValueError(f"No synchronized frames remain for {episode.name}")

    return {
        "frame_idx": frame_idx.astype(np.int64),
        "encoder_idx": encoder_idx.astype(np.int64),
        "pose_idx": pose_idx.astype(np.int64),
    }


def check_output_paths(paths: ProcessedPaths, overwrite: bool) -> None:
    existing = [path for path in (paths.video_path, paths.hand_path, paths.eef_path, paths.sync_path) if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Processed files already exist and OVERWRITE_PROCESSED_FILES=False:\n"
            f"{formatted}"
        )


def process_episode(config: PostprocessConfig, episode: RawEpisode) -> dict[str, Any] | None:
    required_paths = (
        episode.camera_path,
        episode.camera_timestamps_path,
        episode.midas_path,
        episode.piper_path,
        episode.metadata_path,
    )
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{episode.name} is missing files: {missing}")

    metadata = read_json(episode.metadata_path)
    status = str(metadata.get("status", ""))
    if config.process_only_raw_statuses and status not in config.process_only_raw_statuses:
        print(f"Skipping {episode.name}: raw status is {status!r}")
        return None

    out = processed_paths(config.dataset_root, episode.index)
    check_output_paths(out, config.overwrite_processed_files)

    frame_idx, camera_ns = load_camera_timestamps(episode)
    hand, hand_ns = load_midas(episode)
    eef, eef_ns = load_piper(episode)
    sync = make_sync(config, frame_idx, camera_ns, hand_ns, eef_ns, episode)

    shutil.copy2(episode.camera_path, out.video_path)
    np.save(out.hand_path, hand)
    np.save(out.eef_path, eef)
    np.savez_compressed(out.sync_path, **sync)

    summary = {
        "episode": episode.name,
        "index": episode.index,
        "camera_frames": int(frame_idx.size),
        "hand_samples": int(hand.shape[0]),
        "eef_samples": int(eef.shape[0]),
        "sync_rows": int(sync["frame_idx"].size),
        "video": str(out.video_path),
        "hand": str(out.hand_path),
        "eef": str(out.eef_path),
        "sync": str(out.sync_path),
    }
    print(
        f"Processed {episode.name}: {summary['sync_rows']} sync rows, "
        f"{summary['hand_samples']} hand samples, {summary['eef_samples']} EEF samples."
    )
    return summary


def import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing OpenCV Python bindings. Install with: python3 -m pip install opencv-python"
        ) from exc
    return cv2


def validate_processed_dataset(dataset_root: Path) -> None:
    cv2 = import_cv2()
    videos = sorted((dataset_root / "camera" / "mp4_files").glob("*.mp4"))
    hands = sorted((dataset_root / "processed_encoder_t").glob("*.npy"))
    eefs = sorted((dataset_root / "absolute_action_t").glob("*.npy"))
    syncs = sorted((dataset_root / "syncs").glob("*.npz"))
    if not (len(videos) == len(hands) == len(eefs) == len(syncs) > 0):
        raise ValueError(
            "Processed folders must contain the same nonzero number of files: "
            f"videos={len(videos)}, hands={len(hands)}, eefs={len(eefs)}, syncs={len(syncs)}"
        )

    for video_path, hand_path, eef_path, sync_path in zip(videos, hands, eefs, syncs):
        if not (
            video_path.stem == hand_path.stem == eef_path.stem == sync_path.stem
        ):
            raise ValueError(
                "Processed filenames do not align: "
                f"{video_path.name}, {hand_path.name}, {eef_path.name}, {sync_path.name}"
            )
        hand = np.load(hand_path)
        eef = np.load(eef_path)
        sync = np.load(sync_path)
        if hand.ndim != 2 or hand.shape[1] != 13:
            raise ValueError(f"{hand_path} has invalid shape {hand.shape}")
        if eef.ndim != 2 or eef.shape[1] != 6:
            raise ValueError(f"{eef_path} has invalid shape {eef.shape}")
        if not np.isfinite(hand).all():
            raise ValueError(f"{hand_path} contains NaN or Inf")
        if not np.isfinite(eef).all():
            raise ValueError(f"{eef_path} contains NaN or Inf")

        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frame_count <= 0:
            raise ValueError(f"{video_path} has no readable frames")

        for key in ("frame_idx", "encoder_idx", "pose_idx"):
            if key not in sync:
                raise ValueError(f"{sync_path} is missing {key!r}")
        frame_idx = sync["frame_idx"].astype(np.int64)
        encoder_idx = sync["encoder_idx"].astype(np.int64)
        pose_idx = sync["pose_idx"].astype(np.int64)
        if not (frame_idx.ndim == encoder_idx.ndim == pose_idx.ndim == 1):
            raise ValueError(f"{sync_path} sync arrays must be 1D")
        if not (len(frame_idx) == len(encoder_idx) == len(pose_idx) > 0):
            raise ValueError(f"{sync_path} sync arrays must have the same nonzero length")
        if frame_idx.min() < 0 or frame_idx.max() >= frame_count:
            raise ValueError(f"{sync_path} has frame_idx out of range")
        if encoder_idx.min() < 0 or encoder_idx.max() >= len(hand):
            raise ValueError(f"{sync_path} has encoder_idx out of range")
        if pose_idx.min() < 0 or pose_idx.max() >= len(eef):
            raise ValueError(f"{sync_path} has pose_idx out of range")


def main() -> int:
    config = build_config()
    validate_config(config)
    ensure_processed_dirs(config.dataset_root)

    summaries = []
    for episode in discover_raw_episodes(config):
        summary = process_episode(config, episode)
        if summary is not None:
            summaries.append(summary)
    if not summaries:
        raise RuntimeError(f"No raw episodes were processed from {config.raw_dir}")

    summary_path = config.dataset_root / "processing_summary.json"
    summary_path.write_text(
        json.dumps({"episodes": summaries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_processed_dataset(config.dataset_root)
    print(f"Validated {len(summaries)} processed episodes under {config.dataset_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
