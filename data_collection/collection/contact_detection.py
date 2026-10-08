from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np


# PIP, MCP, and AB/AD joints for fingers 1, 2, and 3.
CONTACT_MOTOR_IDS = tuple(range(4, 13))
CONTACT_ENTER_THRESHOLD = 0.05  # rad
CONTACT_EXIT_THRESHOLD = 0.03  # rad
# Compatibility name for callers that use the contact-entry threshold.
CONTACT_THRESHOLD = CONTACT_ENTER_THRESHOLD
CONTACT_MAX_VELOCITY_RAD_S = 0.7
CONTACT_MOTION_CURRENT_THRESHOLD_MA = 250.0  # absolute measured motor current
CONTACT_VIRTUAL_SPRING_STIFFNESS = 1.0  # feedback units / rad
CONTACT_SETTLE_TIME_S = 0.05
CONTACT_PERSISTENCE_TIME_S = 0.01  # continuous time above entry threshold
CONTACT_MIN_ENGAGED_TIME_S = 0.10
CONTACT_EXIT_PERSISTENCE_TIME_S = 0.05
# Candidate feedback is measured beyond the exit threshold, so this gap is
# the equivalent candidate value required to cross the entry threshold.
CONTACT_ENTRY_FEEDBACK_THRESHOLD = (
    CONTACT_VIRTUAL_SPRING_STIFFNESS
    * (CONTACT_ENTER_THRESHOLD - CONTACT_EXIT_THRESHOLD)
)


class ContactState(Enum):
    MOVING = "moving"
    SUPPRESSED = "suppressed"
    ARMED = "armed"
    CONTACT = "contact"


class ContactStateMachine:
    """Detect contact from motion current or stationary position error."""

    def __init__(
        self,
        *,
        max_velocity_rad_s: float = CONTACT_MAX_VELOCITY_RAD_S,
        motion_current_threshold_ma: float = CONTACT_MOTION_CURRENT_THRESHOLD_MA,
        settle_time_s: float = CONTACT_SETTLE_TIME_S,
        persistence_time_s: float = CONTACT_PERSISTENCE_TIME_S,
        entry_feedback_threshold: float = CONTACT_ENTRY_FEEDBACK_THRESHOLD,
        min_engaged_time_s: float = CONTACT_MIN_ENGAGED_TIME_S,
        exit_persistence_time_s: float = CONTACT_EXIT_PERSISTENCE_TIME_S,
    ) -> None:
        self.max_velocity_rad_s = max_velocity_rad_s
        self.motion_current_threshold_ma = motion_current_threshold_ma
        self.settle_time_s = settle_time_s
        self.persistence_time_s = persistence_time_s
        self.entry_feedback_threshold = entry_feedback_threshold
        self.min_engaged_time_s = min_engaged_time_s
        self.exit_persistence_time_s = exit_persistence_time_s
        self.state = ContactState.MOVING
        self._settle_started_s: float | None = None
        self._contact_started_s: float | None = None
        self._engaged_started_s: float | None = None
        self._exit_started_s: float | None = None
        self._candidate_sign = 0.0
        self._last_feedback = 0.0

    def reset(self) -> None:
        self.state = ContactState.MOVING
        self._settle_started_s = None
        self._contact_started_s = None
        self._engaged_started_s = None
        self._exit_started_s = None
        self._candidate_sign = 0.0
        self._last_feedback = 0.0

    def _continue_or_exit_contact(
        self,
        *,
        candidate_feedback: float,
        now_s: float,
        exit_state: ContactState,
    ) -> float:
        candidate_sign = float(np.sign(candidate_feedback))
        if self._candidate_sign == 0.0 and candidate_sign != 0.0:
            # Current can indicate contact before a position residual develops.
            # Adopt the residual direction as soon as it becomes available.
            self._candidate_sign = candidate_sign
        if candidate_sign != 0.0 and candidate_sign == self._candidate_sign:
            self._exit_started_s = None
            self._last_feedback = float(candidate_feedback)
            return self._last_feedback

        if self._exit_started_s is None:
            self._exit_started_s = now_s
        engaged_time_s = (
            0.0
            if self._engaged_started_s is None
            else now_s - self._engaged_started_s
        )
        exit_time_s = now_s - self._exit_started_s
        if (
            engaged_time_s < self.min_engaged_time_s
            or exit_time_s < self.exit_persistence_time_s
        ):
            # Keep publishing the last force so the leader does not switch modes
            # during a brief contact-error dropout.
            return self._last_feedback

        if exit_state is ContactState.MOVING:
            self.reset()
        else:
            self.state = exit_state
            self._contact_started_s = None
            self._engaged_started_s = None
            self._exit_started_s = None
            self._candidate_sign = 0.0
            self._last_feedback = 0.0
        return 0.0

    def update(
        self,
        *,
        commanded_velocity_rad_s: float,
        measured_velocity_rad_s: float,
        candidate_feedback: float,
        now_s: float,
        measured_current_ma: float = 0.0,
    ) -> float:
        moving = (
            abs(commanded_velocity_rad_s) > self.max_velocity_rad_s
            or abs(measured_velocity_rad_s) > self.max_velocity_rad_s
        )
        if moving:
            if self.state is ContactState.CONTACT:
                return self._continue_or_exit_contact(
                    candidate_feedback=candidate_feedback,
                    now_s=now_s,
                    exit_state=ContactState.MOVING,
                )

            if self.state is not ContactState.MOVING:
                self._contact_started_s = None
                self._candidate_sign = 0.0
            self.state = ContactState.MOVING
            self._settle_started_s = None
            if abs(measured_current_ma) < self.motion_current_threshold_ma:
                self.reset()
                return 0.0

            if self._contact_started_s is None:
                self._contact_started_s = now_s
                self._candidate_sign = float(np.sign(candidate_feedback))
                return 0.0
            if now_s - self._contact_started_s < self.persistence_time_s:
                return 0.0

            self.state = ContactState.CONTACT
            self._engaged_started_s = now_s
            self._exit_started_s = None
            self._last_feedback = float(candidate_feedback)
            return self._last_feedback

        if self.state is ContactState.MOVING:
            self.state = ContactState.SUPPRESSED
            self._settle_started_s = now_s
            self._contact_started_s = None
            self._candidate_sign = 0.0
            return 0.0

        if self.state is ContactState.SUPPRESSED:
            if self._settle_started_s is None:
                self._settle_started_s = now_s
            if now_s - self._settle_started_s < self.settle_time_s:
                return 0.0
            self.state = ContactState.ARMED

        if self.state is ContactState.CONTACT:
            return self._continue_or_exit_contact(
                candidate_feedback=candidate_feedback,
                now_s=now_s,
                exit_state=ContactState.ARMED,
            )

        candidate_sign = float(np.sign(candidate_feedback))
        entry_candidate = (
            abs(candidate_feedback) > self.entry_feedback_threshold
        )
        if not entry_candidate:
            self.state = ContactState.ARMED
            self._contact_started_s = None
            self._candidate_sign = 0.0
            return 0.0

        if self._contact_started_s is None or candidate_sign != self._candidate_sign:
            self._contact_started_s = now_s
            self._candidate_sign = candidate_sign
            return 0.0
        # ARMED must observe uninterrupted, same-direction threshold excess
        # before it may publish feedback and transition to CONTACT.
        if now_s - self._contact_started_s < self.persistence_time_s:
            return 0.0

        self.state = ContactState.CONTACT
        self._engaged_started_s = now_s
        self._exit_started_s = None
        self._last_feedback = float(candidate_feedback)
        return self._last_feedback


def contact_feedback(
    position_errors: Sequence[float],
    thresholds: float | Sequence[float] = CONTACT_THRESHOLD,
) -> np.ndarray:
    """Return signed virtual-spring feedback beyond the contact deadband."""

    errors = np.asarray(position_errors, dtype=float)
    contact_excess = np.maximum(np.abs(errors) - thresholds, 0.0)
    return CONTACT_VIRTUAL_SPRING_STIFFNESS * np.sign(errors) * contact_excess
