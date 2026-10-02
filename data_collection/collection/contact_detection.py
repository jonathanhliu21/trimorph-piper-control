from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np


CONTACT_MOTOR_IDS = (4, 5)
CONTACT_THRESHOLD = 0.05  # rad
CONTACT_MAX_VELOCITY_RAD_S = 1.0
CONTACT_VIRTUAL_SPRING_STIFFNESS = 1.0  # feedback units / rad
CONTACT_SETTLE_TIME_S = 0.1
CONTACT_PERSISTENCE_TIME_S = 0.05  # continuous time above CONTACT_THRESHOLD


class ContactState(Enum):
    MOVING = "moving"
    SUPPRESSED = "suppressed"
    ARMED = "armed"
    CONTACT = "contact"


class ContactStateMachine:
    """Suppress motion and require a sustained static-threshold excess."""

    def __init__(
        self,
        *,
        max_velocity_rad_s: float = CONTACT_MAX_VELOCITY_RAD_S,
        settle_time_s: float = CONTACT_SETTLE_TIME_S,
        persistence_time_s: float = CONTACT_PERSISTENCE_TIME_S,
    ) -> None:
        self.max_velocity_rad_s = max_velocity_rad_s
        self.settle_time_s = settle_time_s
        self.persistence_time_s = persistence_time_s
        self.state = ContactState.MOVING
        self._settle_started_s: float | None = None
        self._contact_started_s: float | None = None
        self._candidate_sign = 0.0

    def reset(self) -> None:
        self.state = ContactState.MOVING
        self._settle_started_s = None
        self._contact_started_s = None
        self._candidate_sign = 0.0

    def update(
        self,
        *,
        commanded_velocity_rad_s: float,
        measured_velocity_rad_s: float,
        candidate_feedback: float,
        now_s: float,
    ) -> float:
        moving = (
            abs(commanded_velocity_rad_s) > self.max_velocity_rad_s
            or abs(measured_velocity_rad_s) > self.max_velocity_rad_s
        )
        if moving:
            self.reset()
            return 0.0

        if self.state is ContactState.MOVING:
            self.state = ContactState.SUPPRESSED
            self._settle_started_s = now_s
            return 0.0

        if self.state is ContactState.SUPPRESSED:
            if self._settle_started_s is None:
                self._settle_started_s = now_s
            if now_s - self._settle_started_s < self.settle_time_s:
                return 0.0
            self.state = ContactState.ARMED

        candidate_sign = float(np.sign(candidate_feedback))
        if candidate_sign == 0.0:
            self.state = ContactState.ARMED
            self._contact_started_s = None
            self._candidate_sign = 0.0
            return 0.0

        if self.state is ContactState.CONTACT:
            if candidate_sign == self._candidate_sign:
                return float(candidate_feedback)
            self.state = ContactState.ARMED
            self._contact_started_s = None

        if self._contact_started_s is None or candidate_sign != self._candidate_sign:
            self._contact_started_s = now_s
            self._candidate_sign = candidate_sign
            return 0.0
        # ARMED must observe uninterrupted, same-direction threshold excess
        # before it may publish feedback and transition to CONTACT.
        if now_s - self._contact_started_s < self.persistence_time_s:
            return 0.0

        self.state = ContactState.CONTACT
        return float(candidate_feedback)


def contact_feedback(
    position_errors: Sequence[float],
    thresholds: float | Sequence[float] = CONTACT_THRESHOLD,
) -> np.ndarray:
    """Return signed virtual-spring feedback beyond the contact deadband."""

    errors = np.asarray(position_errors, dtype=float)
    contact_excess = np.maximum(np.abs(errors) - thresholds, 0.0)
    return CONTACT_VIRTUAL_SPRING_STIFFNESS * np.sign(errors) * contact_excess
