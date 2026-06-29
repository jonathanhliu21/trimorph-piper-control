"""Thread-safe trajectory buffer between cuRobo GPU thread and 120 Hz MIT loop."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class JointWaypoint:
    positions: np.ndarray   # (6,) rad
    velocities: np.ndarray  # (6,) rad/s


class CuRoboTrajectoryBuffer:
    """
    Written by the cuRobo GPU thread (push_trajectory) and drained by the
    120 Hz piper MIT loop (next_waypoint).  A new push replaces the existing
    trajectory from the current read position to avoid discontinuities.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._positions: np.ndarray | None = None   # (N, 6)
        self._velocities: np.ndarray | None = None  # (N, 6)
        self._read_idx: int = 0
        self._total: int = 0

    # ------------------------------------------------------------------
    # Writer API (GPU thread)
    # ------------------------------------------------------------------

    def push_trajectory(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        dt_s: float,  # kept for API completeness; spacing is implicit at 1/output_hz
    ) -> None:
        """Replace the pending trajectory with a freshly planned one."""
        positions = np.asarray(positions, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 6:
            raise ValueError(f"positions must be (N, 6), got {positions.shape}")
        with self._lock:
            self._positions = positions.copy()
            self._velocities = velocities.copy()
            self._read_idx = 0
            self._total = positions.shape[0]

    # ------------------------------------------------------------------
    # Reader API (120 Hz piper loop)
    # ------------------------------------------------------------------

    def next_waypoint(self) -> JointWaypoint | None:
        """Return the next waypoint and advance the index, or None when exhausted."""
        with self._lock:
            if self._positions is None or self._read_idx >= self._total:
                return None
            wp = JointWaypoint(
                positions=self._positions[self._read_idx].copy(),
                velocities=self._velocities[self._read_idx].copy(),
            )
            self._read_idx += 1
            return wp

    def is_stale(self) -> bool:
        """True when the buffer is exhausted and no new trajectory has arrived."""
        with self._lock:
            return self._positions is None or self._read_idx >= self._total

    def reset(self) -> None:
        """Clear the buffer; called on teleop disable or emergency stop."""
        with self._lock:
            self._positions = None
            self._velocities = None
            self._read_idx = 0
            self._total = 0
