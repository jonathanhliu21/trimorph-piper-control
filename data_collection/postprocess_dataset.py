#!/usr/bin/env python3
"""Prepare reference-format arrays and replay CSVs from raw collection episodes.

    python data_collection/postprocess_dataset.py data_trimorph_quest --episode 0
        --calibration-json calibration.json --no-camera

Quest conversion uses relative XYZ and rotation mappings. The start pose is
--piper-start-pose, else the one recorded by collect_hand_only.py
--piper-start-joints-deg; without either the result starts at zero/identity and
is marked relative, and replay anchors this motion at the measured Piper pose.
With --pose-source mocap the calibration's Vicon
marker plate recorded in mocap.csv (collect_hand_only.py --use-mocap) is
converted to the flange pose instead, using the calibration's Vicon-to-Piper
and plate-to-flange transforms. Raw files are preserved.
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
    absolute: bool = False
    max_start_offset_mm: float = 150.0
    pose_source: str = "quest"
    mocap_topic: str | None = None
    max_start_rotation_deg: float = 45.0
    mocap_smooth_s: float = 0.02


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


def convert_quest(episode, calibration, start_pose=None, absolute=False, max_start_offset_mm=150.0):
    if (absolute or tracked_point_offset(calibration) is not None) and start_pose is None:
        raise ValueError("Tracked-point offset calibration and --absolute require --piper-start-pose with the "
                         "actual Piper starting XYZ/roll/pitch/yaw; the correction depends on flange orientation")
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
    start_offset = None
    if absolute:
        # Keep where the controller actually was: p = A q + R l + c with the calibration's origin c,
        # instead of pinning the first sample to the start pose. Orientation stays relative to it.
        from scripts.teleop.quest_replay import calibration_origin

        origin, _spread = calibration_origin(calibration)
        lever = tracked_point_offset(calibration)
        xyz = (poses[:, :3] @ np.asarray(calibration["linear_matrix"], float).T + origin
               + (rotations.apply(lever) if lever is not None else 0.0))
        start_offset = check_start_offset(xyz, start, max_start_offset_mm,
                                          "the Quest tracking frame has probably changed since calibration, "
                                          "so recalibrate")
    converted, eef = replay_rows(episode, rows, stamps, xyz, rotations)
    return converted, eef, stamps, start_offset


def check_start_offset(xyz, start, max_start_offset_mm, cause):
    """Distance of an absolute trajectory's first point from the Piper start pose, within the limit."""
    offset = xyz[0] - start[:3]
    if 1000 * np.linalg.norm(offset) > max_start_offset_mm:
        raise ValueError(
            f"Absolute first pose is {1000 * np.linalg.norm(offset):.0f} mm from the Piper start pose "
            f"(limit {max_start_offset_mm:g} mm, --max-start-offset-mm); {cause} or process without --absolute")
    return offset


def replay_rows(episode, rows, stamps, xyz, rotations):
    """Replay CSV rows (XYZ + unwrapped Euler) and the XYZ/rotation-vector action array."""
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
    return converted, eef


def vicon_to_piper(calibration):
    """Rotation and translation (m) taking Vicon world to the Piper base, from a --reference mocap fit."""
    reference = calibration.get("vicon_reference") or {}
    if not reference:
        raise ValueError("Calibration has no vicon_reference; fit one with "
                         "scripts/calib/fit_quest_calibration.py RUN --reference mocap")
    info = next(iter(reference.values()))
    transform = tuple(np.asarray(info[key], float) for key in (
        "vicon_to_piper_rotation", "vicon_to_piper_translation_m",
        "flange_in_vicon_body_rotation", "flange_in_vicon_body_translation_m"))
    for rotation in transform[0::2]:
        if rotation.shape != (3, 3) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
            raise ValueError("vicon_reference rotations must be proper 3x3 matrices")
    if any(translation.shape != (3,) for translation in transform[1::2]):
        raise ValueError("vicon_reference translations must be 3-vectors")
    return transform


def frame_clock(frames, receipt_ns):
    """Host monotonic_ns for each Vicon frame from its frame number.

    Vicon samples arrive over Wi-Fi in bursts, so receipt times bunch together and make
    smooth motion look like jumps. Frame numbers are evenly spaced; fit their rate and
    anchor them to the earliest arrivals (the lowest-latency deliveries).
    """
    slope, intercept = np.polyfit(frames - frames[0], receipt_ns - receipt_ns[0], 1)
    predicted = slope * (frames - frames[0])
    latency = (receipt_ns - receipt_ns[0]) - predicted
    return (receipt_ns[0] + np.percentile(latency, 5) + predicted).round().astype(np.int64)


def smooth_poses(positions, quaternions, sample_s, smooth_s):
    """Remove single-frame spikes (5-sample median) and jitter (Gaussian, sigma smooth_s); 0 = raw.

    A hand-held 3-marker plate flicks in orientation when a marker is partly hidden, and
    the plate-to-flange offset turns each flick into flange translation.
    """
    from scipy.ndimage import gaussian_filter1d, median_filter

    if smooth_s <= 0:
        return positions, quaternions
    quaternions = quaternions * np.where(np.cumsum(np.r_[0, np.einsum(
        "ij,ij->i", quaternions[1:], quaternions[:-1]) < 0]) % 2, -1, 1)[:, None]  # continuous sign
    positions = median_filter(positions, size=(5, 1), mode="nearest")
    quaternions = median_filter(quaternions, size=(5, 1), mode="nearest")
    positions = gaussian_filter1d(positions, smooth_s / sample_s, axis=0, mode="nearest")
    quaternions = gaussian_filter1d(quaternions, smooth_s / sample_s, axis=0, mode="nearest")
    return positions, quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)


def vicon_flange_poses(transform, positions, object_rotations):
    """Flange positions (m) and rotations in the Piper base for Vicon object poses.

    T_flange = W T_object X: W maps Vicon world to the Piper base and X is the flange
    pose in the object's body frame, both fitted by fit_quest_calibration.py.
    """
    r_world, t_world, r_body, t_body = transform
    matrices = object_rotations.as_matrix()
    flange_positions = np.einsum("nij,j->ni", r_world @ matrices, t_body) + positions @ r_world.T + t_world
    return flange_positions, Rotation.from_matrix(r_world @ matrices @ r_body)


def convert_mocap(episode, calibration, start_pose, topic=None, absolute=False, max_start_offset_mm=150.0,
                  max_start_rotation_deg=45.0, smooth_s=0.02):
    """Vicon object poses (mocap.csv) -> Piper flange targets.

    The object must be the marker plate of the calibration (same body-to-flange offset X),
    so each sample becomes the pose the flange would have if the mount sat on the arm.
    --absolute replays that pose as is (Vicon world is fixed); otherwise the first sample
    is pinned to the start pose and the flange-equivalent motion since then is added.
    """
    if start_pose is None:
        raise ValueError("Mocap conversion requires a Piper start pose (recorded at collection or --piper-start-pose)")
    path = episode.path / "mocap.csv"
    rows = [row for row in base.read_csv_rows(path) if row.get("message_type") == "vicon_receiver/msg/Position"]
    topics = sorted({row["topic"] for row in rows})
    if topic is None:
        if len(topics) != 1:
            raise ValueError(f"Choose the tracked object with --mocap-topic; {path} has pose topics {topics}")
        topic = topics[0]
    rows = [row for row in rows if row["topic"] == topic]
    poses = np.array([[float(row[key]) for key in ("x_trans", "y_trans", "z_trans", "x_rot", "y_rot", "z_rot", "w")]
                      for row in rows]).reshape(-1, 7)
    # Vicon reports a lost object as an all-zero pose.
    valid = np.isfinite(poses).all(axis=1) & (np.abs(poses[:, :3]).sum(axis=1) > 0) \
        & (np.linalg.norm(poses[:, 3:], axis=1) > 0.5)
    stamps = np.array([int(row["monotonic_ns"]) for row in rows], dtype=np.int64)
    frames = np.array([int(row["frame_number"]) if str(row.get("frame_number", "")).strip() else -1 for row in rows])
    use_frames = bool(np.all(frames >= 0))
    order = frames if use_frames else stamps
    keep = valid & np.r_[True, np.diff(order) > 0]
    rows, poses, stamps, frames = [row for row, ok in zip(rows, keep) if ok], poses[keep], stamps[keep], frames[keep]
    if len(rows) < 2:
        raise ValueError(f"Fewer than two valid {topic} poses in {path}")
    if use_frames:
        stamps = frame_clock(frames, stamps)
    base.require_monotonic_ns("Mocap", stamps, path)
    if np.any(np.diff(stamps) <= 0):
        raise ValueError(f"Mocap timestamps must be increasing in {path}")
    start = np.asarray(start_pose, dtype=float)
    if start.shape != (6,) or not np.isfinite(start).all():
        raise ValueError("Piper start pose must be six finite XYZ/roll/pitch/yaw values")
    positions, quaternions = smooth_poses(poses[:, :3] / 1000.0, poses[:, 3:],  # Vicon millimetres
                                          float(np.median(np.diff(stamps))) / 1e9, smooth_s)
    flange_xyz, flange_rotations = vicon_flange_poses(
        vicon_to_piper(calibration), positions, Rotation.from_quat(quaternions))
    start_rotation = Rotation.from_euler("xyz", start[3:])
    start_offset = None
    if absolute:
        xyz, rotations = flange_xyz, flange_rotations
        cause = "start each episode with the mount near the arm's start pose"
        start_offset = check_start_offset(xyz, start, max_start_offset_mm, cause)
        angle = np.degrees((rotations[0] * start_rotation.inv()).magnitude())
        if angle > max_start_rotation_deg:
            raise ValueError(f"Absolute first orientation is {angle:.0f} deg from the Piper start pose (limit "
                             f"{max_start_rotation_deg:g} deg, --max-start-rotation-deg); {cause} "
                             "or process without --absolute")
    else:
        xyz = start[:3] + flange_xyz - flange_xyz[0]
        rotations = flange_rotations * flange_rotations[0].inv() * start_rotation
    steps = 1000 * np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    gaps = np.diff(stamps) / 1e6
    print(f"  {topic}: {len(rows)} valid poses ({np.count_nonzero(~valid)} lost-object samples dropped), "
          f"timed by {'Vicon frame number' if use_frames else 'receipt time'}; longest gap {gaps.max():.0f} ms, "
          f"largest flange step {steps.max():.1f} mm, peak flange speed {np.max(steps / gaps) * 1000:.0f} mm/s")
    converted, eef = replay_rows(episode, rows, stamps, xyz, rotations)
    return converted, eef, stamps, start_offset, topic


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
    start_offset = None
    mocap_topic = None
    # Collection with --piper-start-joints-deg records the arm's measured start pose.
    start_pose = config.piper_start_pose or (metadata.get("piper") or {}).get("start_pose")
    if config.calibration_json is not None:
        calibration = base.read_json(config.calibration_json)
        if config.pose_source == "mocap":
            converted, eef, eef_ns, start_offset, mocap_topic = convert_mocap(
                episode, calibration, start_pose, config.mocap_topic, config.absolute, config.max_start_offset_mm,
                config.max_start_rotation_deg, config.mocap_smooth_s)
        else:
            validate_calibration(calibration)
            converted, eef, eef_ns, start_offset = convert_quest(
                episode, calibration, start_pose, config.absolute, config.max_start_offset_mm)
        pose_frame = "relative_to_episode_start" if start_pose is None else "piper_base"
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
    from_quest = calibration is not None and config.pose_source == "quest"
    quest_first = base.read_csv_rows(episode.path / "quest.csv")[0] if from_quest else None
    processing = dict(
        episode=episode.name, episode_index=episode.index, pose_frame=pose_frame,
        source=("piper.csv" if calibration is None else "quest.csv" if from_quest else "mocap.csv"),
        mocap_topic=mocap_topic,
        replay_orientation="Euler xyz radians: Rz(yaw) Ry(pitch) Rx(roll)",
        action_orientation="rotation vector radians", camera_enabled=not config.no_camera,
        sync_reference="hand" if config.no_camera else "camera",
        max_sync_error_s=config.max_sync_error_s,
        trim_camera_frames=0 if config.no_camera else config.trim_camera_frames,
        piper_start_pose=list(start_pose) if start_pose is not None else None,
        position_mapping=(("absolute_vicon" if config.pose_source == "mocap" else "absolute_calibration_origin")
                          if config.absolute
                          else "relative_to_start_pose" if calibration is not None else None),
        first_pose_offset_from_start_m=start_offset.tolist() if start_offset is not None else None,
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
    if start_offset is not None:
        print(f"  Absolute first pose is {np.round(1000 * start_offset, 1).tolist()} mm "
              f"({1000 * np.linalg.norm(start_offset):.1f} mm) from the Piper start pose.")
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
    parser.add_argument("--absolute", action="store_true",
                        help="Keep the controller's actual position (calibration origin) instead of starting the "
                             "trajectory at the Piper start pose; needs the calibration's Quest tracking session")
    parser.add_argument("--max-start-offset-mm", type=float, default=150.0,
                        help="With --absolute, reject episodes whose first pose is farther than this from the "
                             "Piper start pose (a changed Quest tracking frame; default: %(default)s)")
    parser.add_argument("--max-start-rotation-deg", type=float, default=45.0,
                        help="With --absolute and --pose-source mocap, reject episodes whose first orientation is "
                             "farther than this from the Piper start pose (default: %(default)s)")
    parser.add_argument("--mocap-smooth-s", type=float, default=0.02,
                        help="Gaussian smoothing sigma for Vicon poses after removing single-frame spikes "
                             "(0 = raw poses, no filtering; default: %(default)s)")
    parser.add_argument("--pose-source", choices=("quest", "mocap"), default="quest",
                        help="Demonstration pose stream: quest.csv, or the calibration's Vicon marker plate in mocap.csv "
                             "converted to the flange pose (needs a calibration fitted with --reference mocap)")
    parser.add_argument("--mocap-topic", help="Vicon pose topic of the tracked hand object, e.g. /vicon/Hand/Hand "
                                              "(required when mocap.csv has several pose topics)")
    parser.add_argument("--max-sync-error-s", type=float, default=base.MAX_SYNC_ERROR_S)
    parser.add_argument("--trim-camera-frames", type=int, default=base.TRIM_CAMERA_FRAMES)
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.max_sync_error_s) or args.max_sync_error_s <= 0:
        parser.error("--max-sync-error-s must be finite and positive")
    if args.piper_start_pose is not None and (args.calibration_json is None or not np.isfinite(args.piper_start_pose).all()):
        parser.error("--piper-start-pose requires --calibration-json and finite values")
    if (args.absolute or args.pose_source == "mocap") and args.calibration_json is None:
        parser.error("--absolute and --pose-source mocap require --calibration-json")
    if args.mocap_topic is not None and args.pose_source != "mocap":
        parser.error("--mocap-topic requires --pose-source mocap")
    if not np.isfinite(args.max_start_offset_mm) or args.max_start_offset_mm <= 0:
        parser.error("--max-start-offset-mm must be finite and positive")
    if not np.isfinite(args.mocap_smooth_s) or args.mocap_smooth_s < 0:
        parser.error("--mocap-smooth-s must be finite and non-negative")
    if not np.isfinite(args.max_start_rotation_deg) or args.max_start_rotation_deg <= 0:
        parser.error("--max-start-rotation-deg must be finite and positive")
    root = args.dataset_root.expanduser()
    config = PostprocessConfig(root, root / "raw", args.force, {"complete"}, args.max_sync_error_s,
                               args.trim_camera_frames, args.calibration_json.expanduser() if args.calibration_json else None,
                               args.no_camera, tuple(args.piper_start_pose) if args.piper_start_pose is not None else None,
                               args.absolute, args.max_start_offset_mm, args.pose_source, args.mocap_topic,
                               args.max_start_rotation_deg, args.mocap_smooth_s)
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
