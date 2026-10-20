from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import QUEST_PACKET_FIELDS, RAW_EEF_FIELDS, TRIMORPH_MOTOR_IDS
from .utils import WorkerError, utc_stamp, write_csv, write_json


class EpisodeConfig(Protocol):
    hand_config: Path
    hand_port: str
    hand_measure_rate_hz: float
    active_motors: list[int]


@dataclass(frozen=True)
class EpisodePaths:
    episode_index: int
    episode_dir: Path
    camera_path: Path
    camera_timestamps_path: Path
    leader_hand_path: Path
    follower_path: Path
    piper_path: Path
    quest_path: Path
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
        leader_hand_path=episode_dir / "leader_hand.csv",
        follower_path=episode_dir / "follower.csv",
        piper_path=episode_dir / "piper.csv",
        quest_path=episode_dir / "quest.csv",
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
    def __init__(self, paths: EpisodePaths, config: EpisodeConfig) -> None:
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
        self.leader_hand_rows: list[dict[str, object]] = []
        self.follower_rows: list[dict[str, object]] = []
        self.piper_rows: list[dict[str, object]] = []
        self.quest_rows: list[dict[str, object]] = []
        self.video_done = threading.Event()

    def elapsed_s(self, sample_ns: int) -> float:
        return (sample_ns - self.start_ns) / 1e9

    def append_camera(self, row: dict[str, object]) -> None:
        with self.lock:
            self.camera_rows.append(row)

    def append_leader_hand(self, row: dict[str, object]) -> None:
        with self.lock:
            self.leader_hand_rows.append(row)

    def append_follower(self, row: dict[str, object]) -> None:
        with self.lock:
            self.follower_rows.append(row)

    def append_piper(self, row: dict[str, object]) -> None:
        with self.lock:
            self.piper_rows.append(row)

    def append_quest(self, row: dict[str, object]) -> None:
        with self.lock:
            self.quest_rows.append(row)

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
            leader_hand_rows = list(self.leader_hand_rows)
            follower_rows = list(self.follower_rows)
            piper_rows = list(self.piper_rows)
            quest_rows = list(self.quest_rows)
        write_csv(self.paths.camera_timestamps_path, camera_rows)
        write_csv(self.paths.leader_hand_path, leader_hand_rows)
        if getattr(self.config, "follower_hand_config", None) is not None:
            write_csv(self.paths.follower_path, follower_rows)
        write_csv(self.paths.piper_path, piper_rows)
        if getattr(self.config, "use_quest", False):
            write_csv(self.paths.quest_path, quest_rows)
        write_json(
            self.paths.metadata_path,
            self._metadata(
                camera_rows,
                leader_hand_rows,
                follower_rows,
                piper_rows,
                quest_rows,
            ),
        )

    def _metadata(
        self,
        camera_rows: list[dict[str, object]],
        leader_hand_rows: list[dict[str, object]],
        follower_rows: list[dict[str, object]],
        piper_rows: list[dict[str, object]],
        quest_rows: list[dict[str, object]],
    ) -> dict[str, Any]:
        stop_ns = self.stop_ns or time.monotonic_ns()
        follower_config = getattr(self.config, "follower_hand_config", None)
        follower_port = getattr(self.config, "follower_hand_port", None)
        has_camera = hasattr(self.config, "task_camera_width") and bool(
            getattr(self.config, "add_camera", True)
        )
        has_piper = hasattr(self.config, "piper_rate_hz") and bool(
            getattr(self.config, "use_piper", True)
        )
        has_quest = bool(getattr(self.config, "use_quest", False))
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
                "leader_hand_csv": self.paths.leader_hand_path.name,
                "follower_csv": (
                    self.paths.follower_path.name
                    if follower_config is not None
                    else None
                ),
                "piper_csv": self.paths.piper_path.name,
                "quest_csv": self.paths.quest_path.name if has_quest else None,
            },
            "counts": {
                "camera_frames": len(camera_rows),
                "leader_hand_samples": len(leader_hand_rows),
                "follower_samples": len(follower_rows),
                "piper_samples": len(piper_rows),
                "quest_samples": len(quest_rows),
            },
            "camera": {
                "device": "oak1_depthai" if has_camera else None,
                "width": (
                    getattr(self.config, "task_camera_width", None)
                    if has_camera
                    else None
                ),
                "height": (
                    getattr(self.config, "task_camera_height", None)
                    if has_camera
                    else None
                ),
                "fps": (
                    getattr(self.config, "task_camera_fps", None)
                    if has_camera
                    else None
                ),
                "flip": (
                    getattr(self.config, "task_camera_flip", None)
                    if has_camera
                    else None
                ),
                "brightness": (
                    getattr(self.config, "task_camera_brightness", None)
                    if has_camera
                    else None
                ),
                "anti_banding_hz": (
                    getattr(self.config, "task_camera_anti_banding_hz", None)
                    if has_camera
                    else None
                ),
            },
            "trimorph": {
                "motor_ids": list(TRIMORPH_MOTOR_IDS),
                "active_motors": list(self.config.active_motors),
                "hand": _hand_metadata(self.config.hand_config, self.config.hand_port),
                "measure_rate_hz": self.config.hand_measure_rate_hz,
                "force_feedback_rate_hz": getattr(
                    self.config, "force_feedback_rate_hz", None
                ),
                "disabled_force_feedback_motor_ids": list(
                    getattr(self.config, "disabled_force_feedback_motor_ids", ())
                ),
                "training_uses": "measured positions from leader_hand.csv",
            },
            "follower": {
                "enabled": follower_config is not None,
                "hand": (
                    _hand_metadata(follower_config, follower_port)
                    if follower_config is not None and follower_port is not None
                    else None
                ),
                "measure_rate_hz": getattr(self.config, "follower_measure_rate_hz", None),
                "command_rate_hz": getattr(self.config, "follower_command_rate_hz", None),
                "command_scale": getattr(self.config, "follower_command_scale", None),
                "joint_command_scales": getattr(
                    self.config, "follower_joint_command_scales", None
                ),
                "joint_command_offsets": getattr(
                    self.config, "follower_joint_command_offsets", None
                ),
                "thumb_retarget_model": (
                    str(self.config.thumb_retarget_model)
                    if getattr(self.config, "thumb_retarget_model", None) is not None
                    else None
                ),
                "interpolation_alpha": getattr(
                    self.config, "follower_interpolation_alpha", None
                ),
                "max_step_rad": getattr(self.config, "follower_max_step_rad", None),
            },
            "piper": {
                "mode": "passive_drag_teach_logging" if has_piper else None,
                "joint_level_recording": (
                    bool(getattr(self.config, "piper_joint_level", False))
                    if has_piper
                    else False
                ),
                "target_rate_hz": (
                    getattr(self.config, "piper_rate_hz", None)
                    if has_piper
                    else None
                ),
                "raw_eef_fields": list(RAW_EEF_FIELDS),
                "raw_joint_fields": (
                    [f"joint_{joint}" for joint in range(1, 7)]
                    if has_piper and getattr(self.config, "piper_joint_level", False)
                    else []
                ),
            },
            "quest": {
                "mode": "udp_pose_logging" if has_quest else None,
                "bind_host": (
                    getattr(self.config, "quest_host", None) if has_quest else None
                ),
                "port": (
                    getattr(self.config, "quest_port", None) if has_quest else None
                ),
                "startup_timeout_s": (
                    getattr(self.config, "quest_startup_timeout_s", None)
                    if has_quest
                    else None
                ),
                "raw_fields": list(QUEST_PACKET_FIELDS) if has_quest else [],
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
