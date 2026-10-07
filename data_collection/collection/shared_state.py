from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FollowerContactSnapshot:
    motor_ids: tuple[int, ...]
    contact_feedback: tuple[float, ...]
    normalized_joint_difference: tuple[float, ...]


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_hand_target: np.ndarray | None = None
        self._latest_follower_position_error: np.ndarray | None = None
        self._latest_follower_motor_ids: tuple[int, ...] | None = None
        self._latest_follower_contact_snapshot: FollowerContactSnapshot | None = None
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

    def set_follower_position_error(
        self,
        position_error: np.ndarray,
        motor_ids: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        """Publish threshold-adjusted follower error in follower motor order."""

        value = np.asarray(position_error, dtype=np.float64)
        ids = (
            None
            if motor_ids is None
            else tuple(int(motor_id) for motor_id in motor_ids)
        )
        if ids is not None and len(ids) != len(value):
            raise ValueError("Follower motor IDs and position error must align")
        with self._lock:
            self._latest_follower_position_error = value.copy()
            self._latest_follower_motor_ids = ids

    def latest_follower_position_error(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_follower_position_error is None:
                return None
            return self._latest_follower_position_error.copy()

    def latest_follower_position_error_by_motor_id(self) -> dict[int, float]:
        with self._lock:
            if (
                self._latest_follower_position_error is None
                or self._latest_follower_motor_ids is None
            ):
                return {}
            return {
                motor_id: float(error)
                for motor_id, error in zip(
                    self._latest_follower_motor_ids,
                    self._latest_follower_position_error,
                )
            }

    def set_follower_contact_snapshot(
        self,
        motor_ids: tuple[int, ...] | list[int],
        contact_feedback: np.ndarray | tuple[float, ...] | list[float],
        normalized_joint_difference: np.ndarray | tuple[float, ...] | list[float],
    ) -> None:
        ids = tuple(int(motor_id) for motor_id in motor_ids)
        feedback = tuple(float(value) for value in contact_feedback)
        normalized = tuple(float(value) for value in normalized_joint_difference)
        if not len(ids) == len(feedback) == len(normalized):
            raise ValueError("Follower contact snapshot vectors must align")
        snapshot = FollowerContactSnapshot(ids, feedback, normalized)
        with self._lock:
            self._latest_follower_contact_snapshot = snapshot

    def latest_follower_contact_snapshot(
        self,
    ) -> FollowerContactSnapshot | None:
        with self._lock:
            return self._latest_follower_contact_snapshot

    def set_task_frame(self, bgr: np.ndarray) -> None:
        with self._lock:
            self._latest_task_bgr = bgr.copy()

    def latest_task_frame(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_task_bgr is None:
                return None
            return self._latest_task_bgr.copy()
