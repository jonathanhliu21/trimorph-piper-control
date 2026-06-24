from __future__ import annotations

import math
from typing import Any

import numpy as np


def matrix_to_euler_xyz(r: np.ndarray) -> tuple[float, float, float]:
    # fixed-axis XYZ: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    pitch = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    cp = math.cos(pitch)
    if abs(cp) < 1e-9:
        roll = 0.0
        yaw = math.atan2(float(-r[0, 1]), float(r[1, 1]))
    else:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return roll, pitch, yaw


def landmark_xy_pixels(hand_frame: Any, width: int, height: int) -> np.ndarray | None:
    # `HandLandmarkFrame.landmarks` are wrist-centered 3D landmarks; pixels live here.
    landmarks = getattr(hand_frame, "image_landmarks", None)
    if landmarks is None:
        return None
    try:
        if hasattr(landmarks, "landmark"):
            landmarks = getattr(landmarks, "landmark")
        raw = []
        for lm in landmarks:
            if hasattr(lm, "x") and hasattr(lm, "y"):
                x = float(lm.x)
                y = float(lm.y)
            else:
                x = float(lm[0])
                y = float(lm[1])
            raw.append([x, y])
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        if not np.isfinite(arr).all():
            return None

        x = arr[:, 0]
        y = arr[:, 1]
        normalized = (
            np.all((x >= -0.25) & (x <= 1.25))
            and np.all((y >= -0.25) & (y <= 1.25))
        )
        if normalized:
            return np.stack([x * width, y * height], axis=1)
        return np.stack([x, y], axis=1)
    except Exception:
        return None


def as_xyz_array(points: Any) -> np.ndarray | None:
    if points is None:
        return None
    try:
        rows = []
        for p in points:
            if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
                x = float(p.x)
                y = float(p.y)
                z = float(p.z)
            else:
                x = float(p[0])
                y = float(p[1])
                z = float(p[2])
            rows.append([x, y, z])
        arr = np.asarray(rows, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None
        if not np.isfinite(arr).all():
            return None
        return arr
    except Exception:
        return None


def camera_points_from_depth(
    pixels_xy: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])
    h, w = depth_m.shape[:2]
    pts = []
    valid = []
    for u, v in pixels_xy:
        ui = int(round(float(u)))
        vi = int(round(float(v)))
        if ui < 0 or vi < 0 or ui >= w or vi >= h:
            valid.append(False)
            pts.append([math.nan, math.nan, math.nan])
            continue
        z = float(depth_m[vi, ui])
        if not math.isfinite(z) or z <= 0:
            valid.append(False)
            pts.append([math.nan, math.nan, math.nan])
            continue
        x = (float(ui) - ppx) / fx * z
        y = (float(vi) - ppy) / fy * z
        pts.append([x, y, z])
        valid.append(True)
    return np.asarray(pts, dtype=np.float64), np.asarray(valid, dtype=bool)


def rigid_transform(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or a.shape[0] < 4:
        return None
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    aa = a - a_mean
    bb = b - b_mean
    h = aa.T @ bb
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = b_mean - r @ a_mean
    return r, t


def estimate_wrist_pose_cam(
    cv2_module: Any,
    hand_frame: Any,
    teleop_landmarks: Any,
    depth_m: np.ndarray,
    intrinsics: dict[str, float],
    *,
    width: int,
    height: int,
) -> np.ndarray | None:
    pixels_xy = landmark_xy_pixels(hand_frame, width, height)
    wrist_points = as_xyz_array(teleop_landmarks)
    if pixels_xy is None or wrist_points is None or len(pixels_xy) != len(wrist_points):
        return None

    cam_points, depth_valid = camera_points_from_depth(pixels_xy, depth_m, intrinsics)
    finite_wrist = np.all(np.isfinite(wrist_points), axis=1)
    depth_ok = depth_valid & np.all(np.isfinite(cam_points), axis=1) & finite_wrist
    if int(np.count_nonzero(depth_ok)) >= 6:
        rt = rigid_transform(wrist_points[depth_ok], cam_points[depth_ok])
        if rt is not None:
            r, t = rt
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = r
            pose[:3, 3] = t
            return pose

    pnp_ok = finite_wrist & np.all(np.isfinite(pixels_xy), axis=1)
    if int(np.count_nonzero(pnp_ok)) < 6:
        return None

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])
    camera_matrix = np.asarray(
        [[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    success = False
    try:
        success, rvec, tvec, _inliers = cv2_module.solvePnPRansac(
            wrist_points[pnp_ok].astype(np.float64, copy=False),
            pixels_xy[pnp_ok].astype(np.float64, copy=False),
            camera_matrix,
            distCoeffs=dist,
            flags=cv2_module.SOLVEPNP_EPNP,
            reprojectionError=6.0,
            confidence=0.99,
            iterationsCount=100,
        )
    except Exception:
        success = False
    if not success:
        return None

    rmat, _ = cv2_module.Rodrigues(rvec)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rmat, dtype=np.float64)
    pose[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return pose
