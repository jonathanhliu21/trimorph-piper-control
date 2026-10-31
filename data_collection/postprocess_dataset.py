#!/usr/bin/env python3
"""Prepare reference-format arrays and replay CSVs from raw collection episodes.

    python data_collection/postprocess_dataset.py data_trimorph_quest --episode 0
        --calibration-json calibration.json --no-camera

Quest conversion uses relative XYZ and rotation mappings. Without an explicit
--piper-start-pose the result starts at zero/identity and is marked relative;
replay anchors this motion at the measured Piper pose. Raw files are preserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.calib.get_coord_transform import tracked_point_offset, transform_chunk
from scripts.dataset import postprocess_dataset as base


@dataclass(frozen=True)
class PostprocessConfig(base.PostprocessConfig):
    calibration_json: Path | None = None
    no_camera: bool = False
    piper_start_pose: tuple[float, ...] | None = None


def processed_episode_dir(episode_dir: Path) -> Path:
    if episode_dir.parent.name != "raw":
        raise ValueError(f"Expected an episode under DATASET/raw: {episode_dir}")
    return episode_dir.parent.parent / "processed" / episode_dir.name


def validate_calibration(calibration):
    if calibration.get("direction") != "quest_to_piper":
        raise ValueError("Calibration must have direction quest_to_piper")
    a = np.asarray(calibration.get("linear_matrix"), dtype=float)
    b = np.asarray(calibration.get("orientation", {}).get("rotation_matrix"), dtype=float)
    if a.shape != (3, 3) or not np.isfinite(a).all() or np.linalg.matrix_rank(a) < 3:
        raise ValueError("Calibration needs a finite, nonsingular linear_matrix")
    if (b.shape != (3, 3) or not np.isfinite(b).all()
            or not np.allclose(b.T @ b, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(b), 1., atol=1e-6)):
        raise ValueError("Calibration needs a proper orientation.rotation_matrix")
    tracked_point_offset(calibration)


def convert_quest(episode, calibration, start_pose=None):
    if tracked_point_offset(calibration) is not None and start_pose is None:
        raise ValueError("Tracked-point offset calibration requires --piper-start-pose with the actual "
                         "Piper starting XYZ/roll/pitch/yaw; the correction depends on flange orientation")
    path = episode.path / "quest.csv"
    rows = base.read_csv_rows(path)
    if not rows:
        raise ValueError(f"No Quest samples in {path}")
    stamps = base.int_column(rows, "monotonic_ns", path)
    base.require_monotonic_ns("Quest", stamps, path)
    if np.any(np.diff(stamps) <= 0):
        raise ValueError(f"Quest timestamps must be strictly increasing in {path}")
    poses = np.array([[float(row[key]) for key in ("px", "py", "pz", "qx", "qy", "qz", "qw")]
                      for row in rows])
    if not np.isfinite(poses).all():
        raise ValueError(f"Nonfinite Quest pose in {path}")
    start = np.zeros(6) if start_pose is None else np.asarray(start_pose, dtype=float)
    if start.shape != (6,) or not np.isfinite(start).all():
        raise ValueError("Piper start pose must be six finite XYZ/roll/pitch/yaw values")
    xyz, quats = transform_chunk(poses[:, :3], poses[:, 3:], start[:3],
                                Rotation.from_euler("xyz", start[3:]).as_quat(), calibration)
    rotations = Rotation.from_quat(quats)
    # Equivalent continuous Euler representations reduce +/-pi replay discontinuities.
    rpy = np.unwrap(rotations.as_euler("xyz"), axis=0)
    metadata = base.read_json(episode.metadata_path)
    origin = int(metadata["started_monotonic_ns"])
    converted = []
    for i, (raw, stamp, position, angles) in enumerate(zip(rows, stamps, xyz, rpy)):
        row = dict(sample_idx=i, monotonic_ns=int(stamp),
                   elapsed_s=f"{(int(stamp) - origin) / 1e9:.9f}")
        if "wall_time_s" in raw:
            row["wall_time_s"] = raw["wall_time_s"]
        row.update({key: float(value) for key, value in zip(base.RAW_EEF_FIELDS, [*position, *angles])})
        converted.append(row)
    eef = np.column_stack((xyz, rotations.as_rotvec())).astype(np.float32)
    return converted, eef, stamps


def validate_outputs(out, replay_csv, no_camera):
    hand = np.load(out.hand_path, allow_pickle=False)
    eef = np.load(out.eef_path, allow_pickle=False)
    if hand.ndim != 2 or hand.shape[1] != 13 or not np.isfinite(hand).all():
        raise ValueError(f"Invalid hand array: {out.hand_path}")
    if eef.ndim != 2 or eef.shape[1] != 6 or not np.isfinite(eef).all():
        raise ValueError(f"Invalid XYZ/rotvec array: {out.eef_path}")
    if len(base.read_csv_rows(replay_csv)) != len(eef):
        raise ValueError("Replay CSV and action array sample counts differ")
    with np.load(out.sync_path, allow_pickle=False) as sync:
        keys = ("encoder_idx", "pose_idx") if no_camera else ("frame_idx", "encoder_idx", "pose_idx")
        arrays = [sync[key] for key in keys]
        if not arrays or any(a.ndim != 1 or len(a) != len(arrays[0]) or not len(a) for a in arrays):
            raise ValueError("Sync arrays must have matching nonempty 1D shapes")
        for key, length in (("encoder_idx", len(hand)), ("pose_idx", len(eef))):
            if sync[key].min() < 0 or sync[key].max() >= length:
                raise ValueError(f"Sync {key} out of range")
        if not no_camera:
            cv2 = base.import_cv2()
            cap = cv2.VideoCapture(str(out.video_path))
            try:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                cap.release()
            if count <= 0 or sync["frame_idx"].min() < 0 or sync["frame_idx"].max() >= count:
                raise ValueError(f"Unreadable video or frame indices out of range: {out.video_path}")


def process_episode(config, episode):
    metadata = base.read_json(episode.metadata_path)
    if config.process_only_raw_statuses and metadata.get("status") not in config.process_only_raw_statuses:
        print(f"Skipping {episode.name}: status={metadata.get('status')!r}")
        return None
    out = base.processed_paths(config.dataset_root, episode.index)
    processed = processed_episode_dir(episode.path)
    replay_csv = processed / "piper.csv"
    processing_path = processed / "metadata.json"
    outputs = [out.hand_path, out.eef_path, out.sync_path, replay_csv, processing_path]
    if not config.no_camera:
        outputs.append(out.video_path)
    if not config.overwrite_processed_files:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Outputs already exist; use -f to overwrite: {existing}")
    hand, hand_ns = base.load_leader_hand(episode)
    calibration = None
    if config.calibration_json is not None:
        calibration = base.read_json(config.calibration_json)
        validate_calibration(calibration)
        converted, eef, eef_ns = convert_quest(episode, calibration, config.piper_start_pose)
        pose_frame = "relative_to_episode_start" if config.piper_start_pose is None else "piper_base"
    else:
        eef, eef_ns = base.load_piper(episode)
        converted = base.read_csv_rows(episode.piper_path)
        pose_frame = "piper_base"
    if config.no_camera:
        # No synthetic camera frames: the hand clock is the synchronization reference.
        pose_idx, delta = base.nearest_indices(eef_ns, hand_ns)
        sync = dict(encoder_idx=np.arange(len(hand_ns), dtype=np.int64), pose_idx=pose_idx,
                    monotonic_ns=hand_ns)
        bad = np.count_nonzero(delta > round(config.max_sync_error_s * 1e9))
        if bad:
            print(f"Warning: {episode.name}: {bad} hand samples exceed sync tolerance; retaining nearest poses.")
        frame_count = 0
    else:
        frame_idx, camera_ns = base.load_camera_timestamps(episode)
        trim = config.trim_camera_frames
        if trim and len(frame_idx) > 2 * trim:
            frame_idx, camera_ns = frame_idx[trim:-trim], camera_ns[trim:-trim]
        sync = base.make_sync(config, frame_idx, camera_ns, hand_ns, eef_ns, episode)
        frame_count = len(frame_idx)
    sync.update(encoder_monotonic_ns=hand_ns, pose_monotonic_ns=eef_ns)
    quest_first = base.read_csv_rows(episode.path / "quest.csv")[0] if calibration is not None else None
    processing = dict(
        episode=episode.name, episode_index=episode.index, pose_frame=pose_frame,
        source="quest.csv" if calibration is not None else "piper.csv",
        replay_orientation="Euler xyz radians: Rz(yaw) Ry(pitch) Rx(roll)",
        action_orientation="rotation vector radians", camera_enabled=not config.no_camera,
        sync_reference="hand" if config.no_camera else "camera",
        max_sync_error_s=config.max_sync_error_s,
        trim_camera_frames=0 if config.no_camera else config.trim_camera_frames,
        piper_start_pose=list(config.piper_start_pose) if config.piper_start_pose is not None else None,
        calibration_json=str(config.calibration_json.resolve()) if config.calibration_json else None,
        calibration_sha256=hashlib.sha256(config.calibration_json.read_bytes()).hexdigest() if config.calibration_json else None,
        quest_start_position=([float(quest_first[key]) for key in ("px", "py", "pz")]
                              if quest_first is not None else None),
        quest_start_quaternion=([float(quest_first[key]) for key in ("qx", "qy", "qz", "qw")]
                                if quest_first is not None else None),
    )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    if not config.no_camera:
        shutil.copy2(episode.camera_path, out.video_path)
    with replay_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(converted[0]))
        writer.writeheader()
        writer.writerows(converted)
    np.save(out.hand_path, hand)
    np.save(out.eef_path, eef)
    np.savez_compressed(out.sync_path, **sync)
    processing_path.write_text(json.dumps(processing, indent=2) + "\n")
    validate_outputs(out, replay_csv, config.no_camera)
    summary = dict(episode=episode.name, index=episode.index, camera_frames=frame_count,
                   hand_samples=len(hand), eef_samples=len(eef), sync_rows=len(sync["pose_idx"]),
                   video=str(out.video_path) if not config.no_camera else None,
                   hand=str(out.hand_path), eef=str(out.eef_path), sync=str(out.sync_path),
                   piper_csv=str(replay_csv), metadata=str(processing_path), pose_frame=pose_frame)
    print(f"Processed {episode.name}: {len(hand)} hand samples, {len(eef)} EEF samples; {pose_frame}.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path, nargs="?", default=Path("data_trimorph"))
    parser.add_argument("--episode", help="Process only this episode index or episode_XXXXXX name")
    parser.add_argument("--calibration-json", "--calibration", type=Path,
                        help="Quest-to-Piper calibration JSON; omit to process recorded raw Piper poses")
    parser.add_argument("--no-camera", action="store_true", help="Skip all camera reads, copies, and validation")
    parser.add_argument("--piper-start-pose", nargs=6, type=float, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                        help="Absolute Piper starting pose (radians); otherwise Quest output is relative")
    parser.add_argument("--max-sync-error-s", type=float, default=base.MAX_SYNC_ERROR_S)
    parser.add_argument("--trim-camera-frames", type=int, default=base.TRIM_CAMERA_FRAMES)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.max_sync_error_s) or args.max_sync_error_s <= 0:
        parser.error("--max-sync-error-s must be finite and positive")
    if args.piper_start_pose is not None and (args.calibration_json is None or not np.isfinite(args.piper_start_pose).all()):
        parser.error("--piper-start-pose requires --calibration-json and finite values")
    root = args.dataset_root.expanduser()
    config = PostprocessConfig(root, root / "raw", args.force, {"complete"}, args.max_sync_error_s,
                               args.trim_camera_frames, args.calibration_json.expanduser() if args.calibration_json else None,
                               args.no_camera, tuple(args.piper_start_pose) if args.piper_start_pose is not None else None)
    base.validate_config(config)
    episodes = base.discover_raw_episodes(config)
    if args.episode is not None:
        name = f"episode_{int(args.episode):06d}" if args.episode.isdigit() else args.episode
        episodes = [episode for episode in episodes if episode.name == name]
    summaries = [summary for episode in episodes if (summary := process_episode(config, episode)) is not None]
    if not summaries:
        raise ValueError("No matching complete episodes were processed")
    summary_path = root / "processing_summary.json"
    # Preserve summaries for other episodes when processing one episode at a time.
    previous = base.read_json(summary_path).get("episodes", []) if summary_path.exists() else []
    merged = {summary["episode"]: summary for summary in previous}
    merged.update({summary["episode"]: summary for summary in summaries})
    summary_path.write_text(json.dumps({"episodes": sorted(merged.values(), key=lambda row: row["index"])}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
