from __future__ import annotations

from collections.abc import Sequence

import numpy as np


CONTACT_MOTOR_IDS = (4, 5)
CONTACT_THRESHOLD = 0.05  # rad
CONTACT_VELOCITY_ERROR_GAIN_S = 0.05  # rad threshold / (rad/s)
CONTACT_MAX_VELOCITY_RAD_S = 0.1
CONTACT_VIRTUAL_SPRING_STIFFNESS = 1.0  # feedback units / rad


def dynamic_contact_threshold(
    commanded_velocities_rad_s: Sequence[float],
) -> np.ndarray:
    """Return a dynamic threshold, disabling contact above the speed cutoff."""

    velocities = np.asarray(commanded_velocities_rad_s, dtype=float)
    absolute_velocities = np.abs(velocities)
    thresholds = (
        CONTACT_THRESHOLD
        + CONTACT_VELOCITY_ERROR_GAIN_S * absolute_velocities
    )
    return np.where(
        absolute_velocities > CONTACT_MAX_VELOCITY_RAD_S,
        np.inf,
        thresholds,
    )


def contact_feedback(
    position_errors: Sequence[float],
    thresholds: float | Sequence[float] = CONTACT_THRESHOLD,
) -> np.ndarray:
    """Return signed virtual-spring feedback beyond the contact deadband."""

    errors = np.asarray(position_errors, dtype=float)
    contact_excess = np.maximum(np.abs(errors) - thresholds, 0.0)
    return CONTACT_VIRTUAL_SPRING_STIFFNESS * np.sign(errors) * contact_excess
