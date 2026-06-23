from __future__ import annotations

import threading

import numpy as np


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._teacher_mode_active = False
        self._teacher_mode_status: object = None
        self._teleop_enabled = False
        self._latest_midas_target: np.ndarray | None = None
        self._latest_realsense_bgr: np.ndarray | None = None
        self._latest_realsense_depth_m: np.ndarray | None = None
        self._latest_realsense_intrinsics: dict[str, float] | None = None
        self._latest_teleop_bgr: np.ndarray | None = None
        self._latest_wrist_pose_cam: np.ndarray | None = None
        self._stop_episode_reason: str | None = None
        self.calibrate_neutral = threading.Event()
        self.shutdown = threading.Event()

    def set_teacher_mode(self, active: bool, status: object) -> None:
        with self._lock:
            was_active = self._teacher_mode_active
            self._teacher_mode_active = bool(active)
            self._teacher_mode_status = status
            if not active and was_active:
                self._stop_episode_reason = "teacher_mode_lost"

    def teacher_mode(self) -> tuple[bool, object]:
        with self._lock:
            return self._teacher_mode_active, self._teacher_mode_status

    def set_teleop_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._teleop_enabled = bool(enabled)

    def toggle_teleop_enabled(self) -> bool:
        with self._lock:
            self._teleop_enabled = not self._teleop_enabled
            return self._teleop_enabled

    def teleop_enabled(self) -> bool:
        with self._lock:
            return bool(self._teleop_enabled)

    def set_midas_target(self, target: np.ndarray) -> None:
        with self._lock:
            self._latest_midas_target = np.asarray(target, dtype=np.float64).copy()

    def latest_midas_target(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_midas_target is None:
                return None
            return self._latest_midas_target.copy()

    def set_realsense_frame(self, bgr: np.ndarray) -> None:
        with self._lock:
            self._latest_realsense_bgr = bgr.copy()

    def set_realsense_depth(
        self,
        depth_m: np.ndarray,
        intrinsics: dict[str, float] | None,
    ) -> None:
        with self._lock:
            self._latest_realsense_depth_m = np.asarray(depth_m, dtype=np.float32).copy()
            self._latest_realsense_intrinsics = None if intrinsics is None else dict(intrinsics)

    def set_teleop_frame(self, bgr: np.ndarray) -> None:
        with self._lock:
            self._latest_teleop_bgr = bgr.copy()

    def display_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            realsense = (
                None
                if self._latest_realsense_bgr is None
                else self._latest_realsense_bgr.copy()
            )
            teleop = (
                None if self._latest_teleop_bgr is None else self._latest_teleop_bgr.copy()
            )
            return realsense, teleop

    def latest_realsense_rgbd(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, float] | None]:
        with self._lock:
            bgr = None if self._latest_realsense_bgr is None else self._latest_realsense_bgr.copy()
            depth_m = (
                None
                if self._latest_realsense_depth_m is None
                else self._latest_realsense_depth_m.copy()
            )
            intrinsics = (
                None
                if self._latest_realsense_intrinsics is None
                else dict(self._latest_realsense_intrinsics)
            )
            return bgr, depth_m, intrinsics

    def set_wrist_pose_cam(self, pose: np.ndarray | None) -> None:
        with self._lock:
            self._latest_wrist_pose_cam = None if pose is None else np.asarray(pose, dtype=np.float64).copy()

    def latest_wrist_pose_cam(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_wrist_pose_cam is None:
                return None
            return self._latest_wrist_pose_cam.copy()

    def pop_stop_episode_reason(self) -> str | None:
        with self._lock:
            reason = self._stop_episode_reason
            self._stop_episode_reason = None
            return reason
