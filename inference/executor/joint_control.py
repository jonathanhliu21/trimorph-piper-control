"""Policy flange poses -> Piper joint targets: local damped IK plus joint step/speed caps.

Replaces the firmware's Cartesian move_p for the executor. Each target is solved
from the previous joint target by minimising

    |pose error|^2 + damping^2 * |joint change|^2

where damping is zero while the Jacobian's smallest singular value (rotation
rows scaled by 0.2 m, as in replay_actions) is at least DAMPING_SIGMA, and grows
to DAMPING_SIGMA as it approaches zero: exact IK in well-conditioned poses; near
the singularity a small pose error is accepted instead of a large J2/J3 jump.
The joint change is then scaled down, as a whole, so no joint exceeds its speed
cap, and joints stay inside the SDK limits minus replay's 3-degree margin, with
J5 kept on its side of the wrist singularity (|J5| >= 10 deg).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from data_collection.replay_actions import Limits, jacobian, piper_model

ROTATION_SCALE_M = 0.2  # rotation residual weight, the same 0.2 m length as replay_actions.jacobian
DAMPING_SIGMA = 0.03  # damping starts below this; 0.087 at home_cube1, 0.0145 at the oscillating pose
# deg/s, J1..J6: above the 99th-percentile commanded speeds in well-conditioned poses
# of run 20261104_113450 (18, 36, 76, 44, 20, 44).
DEFAULT_MAX_JOINT_SPEED_DEG_S = (45.0, 45.0, 90.0, 60.0, 45.0, 60.0)


def damping(sigma, sigma_start=DAMPING_SIGMA):
    """0 when sigma >= sigma_start, rising to sigma_start as sigma -> 0."""
    return float(np.sqrt(max(0.0, sigma_start ** 2 - sigma ** 2)))


class JointController:
    def __init__(self, max_speed_deg_s=DEFAULT_MAX_JOINT_SPEED_DEG_S, *, sigma_start=DAMPING_SIGMA,
                 limits=Limits()):
        self.fk, bounds = piper_model()
        margin = np.radians(limits.joint_margin_deg)
        self.lower, self.upper = bounds[:, 0] + margin, bounds[:, 1] - margin
        self.wrist_margin = np.radians(limits.wrist_margin_deg)
        self.max_speed = np.radians(np.asarray(max_speed_deg_s, dtype=float))
        if self.max_speed.shape != (6,) or not np.all(self.max_speed > 0):
            raise ValueError("max_speed_deg_s must be six positive values")
        self.sigma_start = sigma_start
        self.q = None

    def reset(self, joints):
        """Start from measured joints (policy start, after a pedal reset)."""
        self.q = np.clip(np.asarray(joints, dtype=float), self.lower, self.upper)

    def bounds(self):
        lower, upper = self.lower.copy(), self.upper.copy()
        # Keep J5 on its current side of zero, at least the wrist margin away.
        if self.q[4] >= 0:
            lower[4] = max(lower[4], self.wrist_margin)
        else:
            upper[4] = min(upper[4], -self.wrist_margin)
        return lower, upper

    def command(self, pose, dt):
        """Joint target (rad) for a Piper XYZ m + Euler xyz rad pose, and diagnostics."""
        if self.q is None:
            raise RuntimeError("JointController.reset() must be called with measured joints first")
        pose = np.asarray(pose, dtype=float)
        position, rotation = pose[:3], Rotation.from_euler("xyz", pose[3:])
        sigma = float(np.linalg.svd(jacobian(self.fk, self.q), compute_uv=False)[-1])
        lam = damping(sigma, self.sigma_start)
        previous = self.q
        lower, upper = self.bounds()

        def pose_error(q):
            current = self.fk(q)
            return np.r_[current[:3] - position,
                         ROTATION_SCALE_M * (Rotation.from_euler("xyz", current[3:]) * rotation.inv()).as_rotvec()]

        def residual(q):
            return np.r_[pose_error(q), lam * (q - previous)]

        start = np.clip(previous, lower + 1e-9, upper - 1e-9)
        solution = least_squares(residual, start, bounds=(lower, upper), max_nfev=100,
                                 ftol=1e-10, xtol=1e-10, gtol=1e-10).x
        step = solution - previous
        # One scale for all joints keeps the flange moving toward the target.
        scale = min(1.0, float(np.min(self.max_speed * dt / np.maximum(np.abs(step), 1e-12))))
        self.q = previous + scale * step
        error = pose_error(self.q)
        return self.q.copy(), {
            "sigma": sigma, "damping": lam, "step_scale": scale,
            "pos_err_mm": float(np.linalg.norm(error[:3]) * 1000),
            "rot_err_deg": float(np.degrees(np.linalg.norm(error[3:]) / ROTATION_SCALE_M)),
        }
