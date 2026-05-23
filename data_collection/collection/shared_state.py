from __future__ import annotations

import threading

import numpy as np


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._teacher_mode_active = False
        self._teacher_mode_status: object = None
        self._latest_midas_target: np.ndarray | None = None
        self._latest_realsense_bgr: np.ndarray | None = None
        self._latest_teleop_bgr: np.ndarray | None = None
        self._stop_episode_reason: str | None = None
        self.calibrate_neutral = threading.Event()
        self.shutdown = threading.Event()

    def set_teacher_mode(self, active: bool, status: object) -> None:
        with self._lock:
            was_active = self._teacher_mode_active
            self._teacher_mode_active = bool(active)
            self._teacher_mode_status = status
            if not active:
                self._latest_midas_target = None
                if was_active:
                    self._stop_episode_reason = "teacher_mode_lost"

    def teacher_mode(self) -> tuple[bool, object]:
        with self._lock:
            return self._teacher_mode_active, self._teacher_mode_status

    def set_midas_target(self, target: np.ndarray) -> None:
        with self._lock:
            if self._teacher_mode_active:
                self._latest_midas_target = np.asarray(target, dtype=np.float64).copy()

    def latest_midas_target(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_midas_target is None:
                return None
            return self._latest_midas_target.copy()

    def set_realsense_frame(self, bgr: np.ndarray) -> None:
        with self._lock:
            self._latest_realsense_bgr = bgr.copy()

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

    def pop_stop_episode_reason(self) -> str | None:
        with self._lock:
            reason = self._stop_episode_reason
            self._stop_episode_reason = None
            return reason
