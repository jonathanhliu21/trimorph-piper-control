from __future__ import annotations

import threading

import numpy as np


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_hand_target: np.ndarray | None = None
        self._latest_task_bgr: np.ndarray | None = None
        self.shutdown = threading.Event()

    def set_hand_target(self, target: np.ndarray) -> None:
        value = np.asarray(target, dtype=np.float64)
        with self._lock:
            self._latest_hand_target = value.copy()

    def latest_hand_target(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_hand_target is None:
                return None
            return self._latest_hand_target.copy()

    def set_task_frame(self, bgr: np.ndarray) -> None:
        with self._lock:
            self._latest_task_bgr = bgr.copy()

    def latest_task_frame(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_task_bgr is None:
                return None
            return self._latest_task_bgr.copy()
