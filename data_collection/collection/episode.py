from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    CANONICAL_MIDAS_MOTOR_IDS,
    TACTILE_FINGER_ORDER,
    CollectionConfig,
)
from .utils import WorkerError, utc_stamp, write_csv, write_json


@dataclass(frozen=True)
class EpisodePaths:
    episode_index: int
    episode_dir: Path
    camera_path: Path
    camera_timestamps_path: Path
    midas_path: Path
    piper_path: Path
    tactile_path: Path
    metadata_path: Path
    teleop_camera_path: Path
    teleop_timestamps_path: Path
    aux_camera_path: Path  # on-disk name "wrist_camera.mp4" is frozen for dataset compat
    aux_timestamps_path: Path  # on-disk name "wrist_timestamps.csv" is frozen for dataset compat


def raw_root(dataset_root: Path) -> Path:
    return dataset_root / "raw"


def next_episode_index(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in root.glob("episode_*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("episode_")
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices, default=-1) + 1


def episode_paths(dataset_root: Path, episode_index: int) -> EpisodePaths:
    episode_dir = raw_root(dataset_root) / f"episode_{episode_index:06d}"
    return EpisodePaths(
        episode_index=episode_index,
        episode_dir=episode_dir,
        camera_path=episode_dir / "camera.mp4",
        camera_timestamps_path=episode_dir / "camera_timestamps.csv",
        midas_path=episode_dir / "midas.csv",
        piper_path=episode_dir / "piper.csv",
        tactile_path=episode_dir / "tactile.npz",
        metadata_path=episode_dir / "metadata.json",
        teleop_camera_path=episode_dir / "teleop_camera.mp4",
        teleop_timestamps_path=episode_dir / "teleop_timestamps.csv",
        aux_camera_path=episode_dir / "wrist_camera.mp4",
        aux_timestamps_path=episode_dir / "wrist_timestamps.csv",
    )


class EpisodeBuffer:
    def __init__(self, paths: EpisodePaths, config: CollectionConfig) -> None:
        self.paths = paths
        self.config = config
        self.start_ns = time.monotonic_ns()
        self.start_wall_time_s = time.time()
        self.started_utc = utc_stamp()
        self.stop_ns: int | None = None
        self.stop_wall_time_s: float | None = None
        self.stopped_utc: str | None = None
        self.stop_reason = ""
        self.errors: list[dict[str, str]] = []

        self.lock = threading.Lock()
        self.camera_rows: list[dict[str, object]] = []
        self.teleop_camera_rows: list[dict[str, object]] = []
        self.aux_camera_rows: list[dict[str, object]] = []
        self.midas_rows: list[dict[str, object]] = []
        self.piper_rows: list[dict[str, object]] = []
        self.tactile_timestamps: list[tuple[int, float, float]] = []
        self.tactile_frames: dict[str, list[np.ndarray]] = {
            finger: [] for finger in TACTILE_FINGER_ORDER
        }
        self.video_done = threading.Event()
        self.teleop_video_done = threading.Event()
        self.aux_video_done = threading.Event()
        if not config.record_teleop_camera:
            self.teleop_video_done.set()
        if not config.aux_camera_enabled:
            self.aux_video_done.set()

    def elapsed_s(self, sample_ns: int) -> float:
        return (sample_ns - self.start_ns) / 1e9

    def append_camera(self, row: dict[str, object]) -> None:
        with self.lock:
            self.camera_rows.append(row)

    def append_teleop_camera(self, row: dict[str, object]) -> None:
        with self.lock:
            self.teleop_camera_rows.append(row)

    def append_aux_camera(self, row: dict[str, object]) -> None:
        with self.lock:
            self.aux_camera_rows.append(row)

    def append_midas(self, row: dict[str, object]) -> None:
        with self.lock:
            self.midas_rows.append(row)

    def append_piper(self, row: dict[str, object]) -> None:
        with self.lock:
            self.piper_rows.append(row)

    def append_tactile(
        self,
        sample_ns: int,
        wall_time_s: float,
        data: dict[str, np.ndarray],
    ) -> None:
        elapsed_s = self.elapsed_s(sample_ns)
        with self.lock:
            self.tactile_timestamps.append((sample_ns, wall_time_s, elapsed_s))
            for finger in TACTILE_FINGER_ORDER:
                if finger in data:
                    self.tactile_frames[finger].append(
                        np.asarray(data[finger], dtype=np.float32).copy()
                    )

    def mark_stopped(self, reason: str) -> None:
        self.stop_ns = time.monotonic_ns()
        self.stop_wall_time_s = time.time()
        self.stopped_utc = utc_stamp()
        self.stop_reason = reason

    def add_error(self, error: WorkerError) -> None:
        self.errors.append(
            {
                "worker": error.name,
                "message": error.message,
                "traceback": error.traceback_text,
            }
        )

    def flush(self) -> None:
        self.paths.episode_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            camera_rows = list(self.camera_rows)
            teleop_camera_rows = list(self.teleop_camera_rows)
            aux_camera_rows = list(self.aux_camera_rows)
            midas_rows = list(self.midas_rows)
            piper_rows = list(self.piper_rows)
            tactile_timestamps = list(self.tactile_timestamps)
            tactile_frames = {
                finger: list(frames) for finger, frames in self.tactile_frames.items()
            }

        write_csv(self.paths.camera_timestamps_path, camera_rows)
        write_csv(self.paths.midas_path, midas_rows)
        write_csv(self.paths.piper_path, piper_rows)
        if self.config.record_teleop_camera:
            write_csv(self.paths.teleop_timestamps_path, teleop_camera_rows)
        if self.config.aux_camera_enabled:
            write_csv(self.paths.aux_timestamps_path, aux_camera_rows)
        self._write_tactile_npz(tactile_timestamps, tactile_frames)
        write_json(
            self.paths.metadata_path,
            self._metadata(camera_rows, midas_rows, piper_rows, tactile_timestamps, aux_camera_rows),
        )

    def _write_tactile_npz(
        self,
        timestamps: list[tuple[int, float, float]],
        frames: dict[str, list[np.ndarray]],
    ) -> None:
        arrays: dict[str, np.ndarray] = {}
        if timestamps:
            arrays["monotonic_ns"] = np.asarray([row[0] for row in timestamps], dtype=np.int64)
            arrays["wall_time_s"] = np.asarray([row[1] for row in timestamps], dtype=np.float64)
            arrays["elapsed_s"] = np.asarray([row[2] for row in timestamps], dtype=np.float64)
        else:
            arrays["monotonic_ns"] = np.zeros((0,), dtype=np.int64)
            arrays["wall_time_s"] = np.zeros((0,), dtype=np.float64)
            arrays["elapsed_s"] = np.zeros((0,), dtype=np.float64)

        for finger in TACTILE_FINGER_ORDER:
            finger_frames = frames.get(finger, [])
            if finger_frames:
                arrays[finger] = np.stack(finger_frames, axis=0).astype(np.float32)
            else:
                arrays[finger] = np.zeros((0, 0, 3), dtype=np.float32)
        np.savez_compressed(self.paths.tactile_path, **arrays)

    def _metadata(
        self,
        camera_rows: list[dict[str, object]],
        midas_rows: list[dict[str, object]],
        piper_rows: list[dict[str, object]],
        tactile_timestamps: list[tuple[int, float, float]],
        aux_camera_rows: list[dict[str, object]] | None = None,
    ) -> dict[str, Any]:
        stop_ns = self.stop_ns or time.monotonic_ns()
        return {
            "episode_index": self.paths.episode_index,
            "status": "complete" if not self.errors else "error",
            "stop_reason": self.stop_reason,
            "started_utc": self.started_utc,
            "stopped_utc": self.stopped_utc,
            "started_monotonic_ns": self.start_ns,
            "stopped_monotonic_ns": stop_ns,
            "duration_s": (stop_ns - self.start_ns) / 1e9,
            "sync_clock": "host monotonic_ns",
            "paths": {
                "camera_mp4": self.paths.camera_path.name,
                "camera_timestamps_csv": self.paths.camera_timestamps_path.name,
                "midas_csv": self.paths.midas_path.name,
                "piper_csv": self.paths.piper_path.name,
                "tactile_npz": self.paths.tactile_path.name,
                "teleop_camera_mp4": (
                    self.paths.teleop_camera_path.name
                    if self.config.record_teleop_camera
                    else None
                ),
                "wrist_camera_mp4": (
                    self.paths.aux_camera_path.name
                    if self.config.aux_camera_enabled
                    else None
                ),
                "wrist_timestamps_csv": (
                    self.paths.aux_timestamps_path.name
                    if self.config.aux_camera_enabled
                    else None
                ),
            },
            "counts": {
                "camera_frames": len(camera_rows),
                "midas_samples": len(midas_rows),
                "piper_samples": len(piper_rows),
                "tactile_samples": len(tactile_timestamps),
                "wrist_camera_frames": len(aux_camera_rows) if aux_camera_rows is not None else 0,
            },
            "camera": {
                "width": self.config.task_camera_width,
                "height": self.config.task_camera_height,
                "fps": self.config.task_camera_fps,
                "serial": "oak1_depthai",
                "flip": self.config.task_camera_flip,
            },
            "teleop": {
                "camera": self.config.teleop_camera,
                "camera_recorded": self.config.record_teleop_camera,
                "hand_type": self.config.teleop_hand_type,
                "input_hand": self.config.teleop_input_hand,
                "selfie": self.config.teleop_selfie,
                "mirror_input": self.config.teleop_mirror_input,
                "mirror_axis": self.config.teleop_mirror_axis,
            },
            "midas": {
                "measured_source": "MidasHand.read_pos()",
                "target_measure_rate_hz": self.config.midas_measure_rate_hz,
                "command_rate_hz": self.config.midas_command_rate_hz,
                "profile_velocity_rad_s": self.config.midas_profile_velocity_rad_s,
                "profile_acceleration_rad_s2": (
                    self.config.midas_profile_acceleration_rad_s2
                ),
                "motor_ids": list(CANONICAL_MIDAS_MOTOR_IDS),
                "training_uses": "measured positions from midas.csv",
            },
            "piper": {
                "target_rate_hz": self.config.piper_rate_hz,
                "raw_eef_format": "[x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]",
                "processed_eef_format": "[x_m, y_m, z_m, rotvec_x_rad, rotvec_y_rad, rotvec_z_rad]",
                "euler_convention": "fixed-axis XYZ: R = Rz(yaw) * Ry(pitch) * Rx(roll)",
            },
            "tactile": {
                "enabled": self.config.enable_tactile,
                "format": "npz with monotonic_ns, wall_time_s, elapsed_s, thumb, index, middle, ring",
                "units": "Newtons, columns [Fx, Fy, Fz]",
                "target_rate_hz": self.config.tactile_rate_hz,
                "fingers": self.config.tactile_fingers,
            },
            "wrist_camera": {
                "enabled": self.config.aux_camera_enabled,
                "device": "oak1_depthai" if self.config.aux_camera_enabled else None,
                "width": self.config.aux_camera_width if self.config.aux_camera_enabled else None,
                "height": self.config.aux_camera_height if self.config.aux_camera_enabled else None,
                "fps": self.config.aux_camera_fps if self.config.aux_camera_enabled else None,
            },
            "errors": self.errors,
        }


class RecordingManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: EpisodeBuffer | None = None

    def current(self) -> EpisodeBuffer | None:
        with self._lock:
            return self._current

    def is_recording(self) -> bool:
        return self.current() is not None

    def start(self, session: EpisodeBuffer) -> None:
        with self._lock:
            if self._current is not None:
                raise RuntimeError("Recording is already active")
            self._current = session

    def stop(self, reason: str) -> EpisodeBuffer | None:
        with self._lock:
            session = self._current
            self._current = None
        if session is not None:
            session.mark_stopped(reason)
        return session
