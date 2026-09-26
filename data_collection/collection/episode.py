from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RAW_EEF_FIELDS, TRIMORPH_MOTOR_IDS, CollectionConfig
from .utils import WorkerError, utc_stamp, write_csv, write_json


@dataclass(frozen=True)
class EpisodePaths:
    episode_index: int
    episode_dir: Path
    camera_path: Path
    camera_timestamps_path: Path
    proprioception_path: Path
    follower_path: Path
    piper_path: Path
    metadata_path: Path


def raw_root(dataset_root: Path) -> Path:
    return dataset_root / "raw"


def next_episode_index(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in root.glob("episode_*"):
        suffix = path.name.removeprefix("episode_")
        if path.is_dir() and suffix.isdigit():
            indices.append(int(suffix))
    return max(indices, default=-1) + 1


def episode_paths(dataset_root: Path, episode_index: int) -> EpisodePaths:
    episode_dir = raw_root(dataset_root) / f"episode_{episode_index:06d}"
    return EpisodePaths(
        episode_index=episode_index,
        episode_dir=episode_dir,
        camera_path=episode_dir / "camera.mp4",
        camera_timestamps_path=episode_dir / "camera_timestamps.csv",
        proprioception_path=episode_dir / "proprioception.csv",
        follower_path=episode_dir / "follower.csv",
        piper_path=episode_dir / "piper.csv",
        metadata_path=episode_dir / "metadata.json",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hand_metadata(path: Path, port: str) -> dict[str, Any]:
    from midas_hand_api import HandConfig

    hand_config = HandConfig.load(path)
    return {
        "config_path": str(path.resolve()),
        "config_sha256": _sha256(path),
        "port": port,
        "motor_model": hand_config.motor_model,
        "expected_model_number": hand_config.expected_model_number,
        "baudrate": hand_config.baudrate,
        "current_limit_ma": hand_config.goal_current_limit,
        "motor_ids": list(hand_config.motor_ids),
    }


class EpisodeBuffer:
    def __init__(self, paths: EpisodePaths, config: CollectionConfig) -> None:
        self.paths = paths
        self.config = config
        self.start_ns = time.monotonic_ns()
        self.started_utc = utc_stamp()
        self.stop_ns: int | None = None
        self.stopped_utc: str | None = None
        self.stop_reason = ""
        self.errors: list[dict[str, str]] = []
        self.lock = threading.Lock()
        self.camera_rows: list[dict[str, object]] = []
        self.proprioception_rows: list[dict[str, object]] = []
        self.follower_rows: list[dict[str, object]] = []
        self.piper_rows: list[dict[str, object]] = []
        self.video_done = threading.Event()

    def elapsed_s(self, sample_ns: int) -> float:
        return (sample_ns - self.start_ns) / 1e9

    def append_camera(self, row: dict[str, object]) -> None:
        with self.lock:
            self.camera_rows.append(row)

    def append_proprioception(self, row: dict[str, object]) -> None:
        with self.lock:
            self.proprioception_rows.append(row)

    def append_follower(self, row: dict[str, object]) -> None:
        with self.lock:
            self.follower_rows.append(row)

    def append_piper(self, row: dict[str, object]) -> None:
        with self.lock:
            self.piper_rows.append(row)

    def mark_stopped(self, reason: str) -> None:
        self.stop_ns = time.monotonic_ns()
        self.stopped_utc = utc_stamp()
        self.stop_reason = reason

    def add_error(self, error: WorkerError) -> None:
        self.errors.append(
            {"worker": error.name, "message": error.message, "traceback": error.traceback_text}
        )

    def flush(self) -> None:
        self.paths.episode_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            camera_rows = list(self.camera_rows)
            proprioception_rows = list(self.proprioception_rows)
            follower_rows = list(self.follower_rows)
            piper_rows = list(self.piper_rows)
        write_csv(self.paths.camera_timestamps_path, camera_rows)
        write_csv(self.paths.proprioception_path, proprioception_rows)
        if self.config.follower_hand_config is not None:
            write_csv(self.paths.follower_path, follower_rows)
        write_csv(self.paths.piper_path, piper_rows)
        write_json(
            self.paths.metadata_path,
            self._metadata(camera_rows, proprioception_rows, follower_rows, piper_rows),
        )

    def _metadata(
        self,
        camera_rows: list[dict[str, object]],
        proprioception_rows: list[dict[str, object]],
        follower_rows: list[dict[str, object]],
        piper_rows: list[dict[str, object]],
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
                "proprioception_csv": self.paths.proprioception_path.name,
                "follower_csv": (
                    self.paths.follower_path.name
                    if self.config.follower_hand_config is not None
                    else None
                ),
                "piper_csv": self.paths.piper_path.name,
            },
            "counts": {
                "camera_frames": len(camera_rows),
                "proprioception_samples": len(proprioception_rows),
                "follower_samples": len(follower_rows),
                "piper_samples": len(piper_rows),
            },
            "camera": {
                "device": "oak1_depthai",
                "width": self.config.task_camera_width,
                "height": self.config.task_camera_height,
                "fps": self.config.task_camera_fps,
                "flip": self.config.task_camera_flip,
            },
            "trimorph": {
                "motor_ids": list(TRIMORPH_MOTOR_IDS),
                "hand": _hand_metadata(self.config.hand_config, self.config.hand_port),
                "measure_rate_hz": self.config.hand_measure_rate_hz,
                "training_uses": "measured positions from proprioception.csv",
            },
            "follower": {
                "enabled": self.config.follower_hand_config is not None,
                "hand": (
                    _hand_metadata(
                        self.config.follower_hand_config,
                        self.config.follower_hand_port,
                    )
                    if self.config.follower_hand_config is not None
                    and self.config.follower_hand_port is not None
                    else None
                ),
                "measure_rate_hz": self.config.follower_measure_rate_hz,
                "command_rate_hz": self.config.follower_command_rate_hz,
                "command_scale": self.config.follower_command_scale,
                "interpolation_alpha": self.config.follower_interpolation_alpha,
                "max_step_rad": self.config.follower_max_step_rad,
            },
            "piper": {
                "mode": "passive_drag_teach_logging",
                "target_rate_hz": self.config.piper_rate_hz,
                "raw_eef_fields": list(RAW_EEF_FIELDS),
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
