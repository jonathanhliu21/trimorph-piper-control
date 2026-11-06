"""Offline start-pose audit; writes plots, per-episode CSVs, and JSON statistics.

Example:
  .venv/bin/python scripts/dataset/analyze_start_positions.py \
      data_trimorph_unscrew_cap_0 --home home_unscrew1 \
      --run output/executor_runs/20261105_153522

No robot connection or configuration changes. Piper joints are IK estimates,
not measured dataset joints. Primary statistics weight each episode once.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trimorph-start-analysis-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_collection.collection.config import HAND_START_POSITIONS_RAD, PIPER_START_JOINTS_DEG
from data_collection.replay_actions import piper_model

CURRENT, PREVIOUS, LIVE = "#d44b27", "#238443", "#843c9d"
BLUE = "#347da8"


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, names, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        w.writerows(rows)


def summary(values, reference=None):
    values = np.asarray(values, dtype=float)
    result = {"mean": values.mean(axis=0).tolist(), "std": values.std(axis=0, ddof=1).tolist()}
    for name, q in (("min", 0), ("p05", 5), ("median", 50), ("p95", 95), ("max", 100)):
        result[name] = np.percentile(values, q, axis=0).tolist()
    if reference is not None:
        result.update(reference=np.asarray(reference).tolist(),
                      reference_minus_median=(reference - np.median(values, axis=0)).tolist(),
                      reference_zscore=((reference - values.mean(axis=0)) / values.std(axis=0, ddof=1)).tolist(),
                      reference_percentile=(100 * np.mean(values <= reference, axis=0)).tolist())
    return result


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--home", default="home_unscrew1")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "output/unscrew_start_analysis")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "font.size": 10})
    fk, bounds = piper_model()
    qhome = np.asarray(PIPER_START_JOINTS_DEG[args.home])
    home = fk(np.radians(qhome))
    # Historical reference explicitly documented in config.py; not an active preset.
    old_q = np.array([-50.73, 108.65, -58.83, 22.08, 36.01, 28.63]) if args.home == "home_unscrew1" else None
    old = fk(np.radians(old_q)) if old_q is not None else None
    hand_home = np.asarray(HAND_START_POSITIONS_RAD[args.home])
    chunks = np.load(args.run / "chunks.npz", allow_pickle=False)
    live = chunks["state"][0].astype(float)
    live_pose = np.r_[live[:3], R.from_rotvec(live[3:6]).as_euler("xyz")]
    episodes, poses, hands, synced_poses, synced_hands, half_poses = [], [], [], [], [], []
    trajectories, deltas, sync_errors, early_changes, pose_gaps = [], [], [], [], []
    sources, calibrations, raw_counts = set(), set(), []
    grid = np.linspace(0, 5, 151)
    for p in sorted((args.dataset / "absolute_action_t").glob("*.npy")):
        idx = int(p.stem)
        pose = np.load(p, allow_pickle=False).astype(float)
        hand = np.load(args.dataset / "processed_encoder_t" / p.name, allow_pickle=False).astype(float)
        z = np.load(args.dataset / "syncs" / f"{p.stem}.npz", allow_pickle=False)
        assert pose.ndim == 2 and pose.shape[1] == 6 and np.isfinite(pose).all()
        assert hand.ndim == 2 and hand.shape[1] == 13 and np.isfinite(hand).all()
        assert str(z["pose_frame"]) == "piper_base"
        for key, size in (("pose_idx", len(pose)), ("encoder_idx", len(hand))):
            assert len(z[key]) and np.min(z[key]) >= 0 and np.max(z[key]) < size
        epdir = args.dataset / "raw" / f"episode_{idx:06d}"
        meta = json.loads((epdir / "metadata.json").read_text())
        pm = json.loads((args.dataset / "processed" / epdir.name / "metadata.json").read_text())
        sources.add(pm["source"])
        calibrations.add(pm["calibration_sha256"])
        raw_counts.append(meta["counts"]["piper_samples"])
        cameras = {int(r["frame_idx"]): int(r["monotonic_ns"]) for r in csv_rows(epdir / "camera_timestamps.csv")}
        t0 = cameras[int(z["frame_idx"][0])]
        pt = (z["pose_monotonic_ns"] - t0) / 1e9
        ht = (z["encoder_monotonic_ns"] - t0) / 1e9
        assert np.all(np.diff(pt) >= 0) and np.all(np.diff(ht) >= 0)
        pi, hi = int(z["pose_idx"][0]), int(z["encoder_idx"][0])
        episodes.append(idx)
        poses.append(pose[0]); hands.append(hand[0])
        synced_poses.append(pose[pi]); synced_hands.append(hand[hi])
        mask = (pt >= 0) & (pt <= .5)
        assert mask.any(), f"No samples in first half-second: {idx}"
        half_poses.append(np.median(pose[mask, :3], axis=0))
        sync_errors.append([pt[pi] * 1000, ht[hi] * 1000])
        pose_gaps.append(np.max(np.diff(pt)[pt[:-1] <= .5]))
        xyz = np.array([np.interp(grid, pt, pose[:, j], left=np.nan, right=np.nan) for j in range(3)]).T
        trajectories.append(xyz * 1000)
        deltas.append((xyz - pose[pi, :3]) * 1000)
        early_changes.append(np.linalg.norm(xyz[15] - pose[pi, :3]) * 1000)
    ids = np.asarray(episodes)
    poses, hands = np.array(poses), np.array(hands)
    sp, sh = np.array(synced_poses), np.array(synced_hands)
    xyz = poses[:, :3] * 1000
    rotations = R.from_rotvec(poses[:, 3:])
    mean_rot = rotations.mean()
    # Unwrap around the mean Euler components, for display only. Distances use SO(3).
    mean_euler = mean_rot.as_euler("xyz", degrees=True)
    angles = mean_euler + (rotations.as_euler("xyz", degrees=True) - mean_euler + 180) % 360 - 180
    home_angles = mean_euler + (np.degrees(home[3:]) - mean_euler + 180) % 360 - 180
    old_angles = None if old is None else mean_euler + (np.degrees(old[3:]) - mean_euler + 180) % 360 - 180
    live_angles = mean_euler + (np.degrees(live_pose[3:]) - mean_euler + 180) % 360 - 180
    rot_dist = np.degrees((rotations * mean_rot.inv()).magnitude())
    home_rot_dist = float(np.degrees((R.from_euler("xyz", home[3:]) * mean_rot.inv()).magnitude()))
    distance = np.linalg.norm(xyz - home[:3] * 1000, axis=1)
    nearest = np.argsort(distance)[:10]

    # Estimate the same IK branch from a fixed seed for every episode.
    joint_estimates, residuals = [], []
    for pose in poses:
        target_rot = R.from_rotvec(pose[3:])
        def residual(q):
            pp = fk(q)
            return np.r_[pp[:3] - pose[:3], .2 * (R.from_euler("xyz", pp[3:]) * target_rot.inv()).as_rotvec()]
        fit = least_squares(residual, np.radians(qhome), bounds=(bounds[:, 0], bounds[:, 1]),
                            max_nfev=150, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        err = residual(fit.x)
        residuals.append([np.linalg.norm(err[:3]) * 1000, np.degrees(np.linalg.norm(err[3:]) / .2)])
        joint_estimates.append(np.degrees(fit.x))
    joints, residuals = np.array(joint_estimates), np.array(residuals)
    valid_ik = (residuals[:, 0] < .5) & (residuals[:, 1] < .2)

    stats = {
        "dataset": str(args.dataset.resolve()), "run": str(args.run.resolve()), "episodes": len(ids),
        "home_name": args.home, "home_joints_deg": qhome.tolist(), "home_fk_xyz_euler": home.tolist(),
        "previous_joints_deg": None if old_q is None else old_q.tolist(),
        "previous_fk_xyz_euler": None if old is None else old.tolist(),
        "source_streams": sorted(sources), "calibration_hashes": sorted(calibrations),
        "raw_piper_samples_total": int(sum(raw_counts)),
        "xyz_mm": summary(xyz, home[:3] * 1000), "euler_deg": summary(angles, home_angles),
        "hand_rad": summary(hands, hand_home), "ik_joints_deg": summary(joints[valid_ik], qhome),
        "rotation_mean_euler_deg": mean_euler.tolist(),
        "rotation_distance_to_mean_deg": summary(rot_dist), "home_rotation_distance_to_mean_deg": home_rot_dist,
        "home_xyz_offset_norm_mm": float(np.linalg.norm(home[:3] * 1000 - np.median(xyz, axis=0))),
        "dataset_xyz_radius_from_median_mm": summary(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1)),
        "distance_from_home_mm": summary(distance),
        "nearest_xyz_episodes": [{"episode": int(ids[i]), "distance_mm": float(distance[i]),
                                  "rotation_to_home_deg": float(np.degrees((rotations[i] * R.from_euler("xyz", home[3:]).inv()).magnitude()))} for i in nearest],
        "live_first_state": live.tolist(), "live_hand_rad": summary(hands, live[6:]),
        "live_minus_home_xyz_mm": ((live[:3] - home[:3]) * 1000).tolist(),
        "first_sync_xyz_mm": summary(sp[:, :3] * 1000, home[:3] * 1000),
        "first_half_second_median_xyz_mm": summary(np.array(half_poses) * 1000, home[:3] * 1000),
        "first_sync_hand_rad": summary(sh, hand_home),
        "first_to_sync_translation_mm": summary(np.linalg.norm(sp[:, :3] - poses[:, :3], axis=1) * 1000),
        "first_to_sync_rotation_deg": summary(np.degrees((R.from_rotvec(sp[:, 3:]) * rotations.inv()).magnitude())),
        "first_sync_timestamp_error_ms_pose_hand": summary(np.array(sync_errors)),
        "first_sync_pose_error_over_10ms_count": int(np.sum(np.abs(np.array(sync_errors)[:, 0]) > 10)),
        "early_max_pose_timestamp_gap_s": summary(pose_gaps),
        "first_half_second_displacement_mm": summary(early_changes),
        "ik_valid_count": int(valid_ik.sum()), "ik_max_residual_mm_deg": residuals.max(axis=0).tolist(),
    }
    stats["pose_neighborhood_counts"] = {}
    for name, reference in (("current", home), ("previous", old)):
        if reference is None:
            continue
        d = np.linalg.norm(xyz - reference[:3]*1000, axis=1)
        a = np.degrees((rotations * R.from_euler("xyz", reference[3:]).inv()).magnitude())
        stats["pose_neighborhood_counts"][name] = {
            "within_20mm_and_10deg": int(np.sum((d <= 20) & (a <= 10))),
            "within_30mm_and_15deg": int(np.sum((d <= 30) & (a <= 15))),
        }
    stats["collection_quarters"] = [
        {"first_episode": int(ids[g[0]]), "last_episode": int(ids[g[-1]]),
         "xyz_median_mm": np.median(xyz[g], axis=0).tolist(),
         "hand_median_rad": np.median(hands[g], axis=0).tolist()}
        for g in np.array_split(np.arange(len(ids)), 4)
    ]
    stats["early_motion_valid_episode_count"] = np.isfinite(np.array(trajectories)[:, :, 0]).sum(axis=0).tolist()
    # Existing plot_run output compares FK feedback to each target at its expected arrival.
    error_path = args.run / "eef_position_error.csv"
    if error_path.exists():
        error_rows = csv_rows(error_path)
        stats["existing_command_arrival_tracking_error_mm"] = {}
        for seconds in (2, 5):
            errors = np.array([float(r["position_error_mm"]) for r in error_rows
                               if float(r["seconds_from_first_observation"]) <= seconds])
            errors = errors[np.isfinite(errors)]
            if len(errors):
                stats["existing_command_arrival_tracking_error_mm"][str(seconds)] = {
                    "samples": len(errors), **summary(errors)}
    write_csv(out / "episode_starts.csv", ["episode", *[f"{s}_mm" for s in "xyz"],
              "roll_deg", "pitch_deg", "yaw_deg", *[f"hand_{i}_rad" for i in range(13)],
              *[f"ik_joint_{i}_deg" for i in range(1, 7)], "ik_position_error_mm", "ik_rotation_error_deg", "distance_from_home_mm"],
              np.column_stack([ids, xyz, angles, hands, joints, residuals, distance]))
    write_csv(out / "synchronized_starts.csv", ["episode", *[f"{s}_mm" for s in "xyz"],
              *[f"hand_{i}_rad" for i in range(13)], "pose_sync_error_ms", "hand_sync_error_ms"],
              np.column_stack([ids, sp[:, :3] * 1000, sh, sync_errors]))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, (i, j) in zip(axs, ((0, 1), (0, 2), (1, 2))):
        sc = ax.scatter(xyz[:, i], xyz[:, j], c=ids, cmap="viridis", s=25, alpha=.8, label="Episode starts")
        ax.scatter(home[i]*1000, home[j]*1000, color=CURRENT, marker="*", s=220, label="Current home", zorder=5)
        if old is not None:
            ax.scatter(old[i]*1000, old[j]*1000, color=PREVIOUS, marker="D", s=70, label="Previous home", zorder=4)
        ax.scatter(live[i]*1000, live[j]*1000, edgecolor=LIVE, facecolor="none", marker="o", s=160, linewidth=2, label="Deployment start", zorder=6)
        ax.set(xlabel=f"{'XYZ'[i]} (mm)", ylabel=f"{'XYZ'[j]} (mm)")
        ax.set_aspect("equal", adjustable="datalim")
    axs[0].legend(fontsize=8)
    fig.colorbar(sc, ax=axs, label="Episode index", shrink=.75)
    fig.suptitle(f"{len(ids)} demonstration starts in Piper base frame | current home is {home[2]*1000-np.median(xyz[:,2]):.1f} mm above median")
    save(fig, out, "01_arm_start_projections")

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    distributions = np.c_[xyz, angles]
    for k, ax in enumerate(axs.flat):
        ax.hist(distributions[:, k], bins=20, color=BLUE, alpha=.75)
        ax.axvspan(*np.percentile(distributions[:, k], [5, 95]), color=BLUE, alpha=.1, label="Dataset 5–95%")
        ax.axvline(np.median(distributions[:, k]), color="black", linestyle=":", label="Dataset median")
        ax.axvline(np.r_[home[:3]*1000, home_angles][k], color=CURRENT, linewidth=2, label="Current home")
        if old is not None:
            ax.axvline(np.r_[old[:3]*1000, old_angles][k], color=PREVIOUS, linestyle="--", label="Previous home")
        ax.axvline(np.r_[live[:3]*1000, live_angles][k], color=LIVE, linestyle=":", label="Deployment start")
        ax.set(xlabel=("X (mm)", "Y (mm)", "Z (mm)", "Roll (deg)", "Pitch (deg)", "Yaw (deg)")[k], ylabel="Episodes")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"Arm start distributions | orientation distance: current home {home_rot_dist:.1f}° from SO(3) mean")
    save(fig, out, "02_arm_start_distributions")

    fig, axs = plt.subplots(4, 4, figsize=(16, 11), layout="constrained")
    fingers = ["Thumb"]*4 + ["Index"]*3 + ["Middle"]*3 + ["Ring"]*3
    for j, ax in enumerate(axs.flat):
        if j >= 13:
            ax.axis("off")
            continue
        ax.hist(hands[:, j], bins=23, color=BLUE, alpha=.75)
        ax.axvline(np.median(hands[:, j]), color="black", linestyle=":", label="Dataset median")
        ax.axvline(hand_home[j], color=CURRENT, linewidth=2, label="Configured home")
        ax.axvline(live[6+j], color=LIVE, linestyle="--", label="Measured deployment start")
        q5, q95 = np.percentile(hands[:, j], [5, 95])
        ax.set(title=f"{fingers[j]} motor {j} | SD {hands[:,j].std(ddof=1):.3f}\n5–95%: [{q5:.3f}, {q95:.3f}]", xlabel="Position (rad)", ylabel="Episodes")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Hand starting joints: first processed encoder row per episode (leader / policy radians)")
    save(fig, out, "03_hand_start_distributions")

    fig, axs = plt.subplots(4, 1, figsize=(14, 11), layout="constrained", sharex=True)
    for j, ax in enumerate(axs[:3]):
        ax.plot(ids, xyz[:, j], ".-", color=BLUE, linewidth=.7, label="Dataset start")
        ax.axhline(home[j]*1000, color=CURRENT, label="Current home")
        if old is not None:
            ax.axhline(old[j]*1000, color=PREVIOUS, linestyle="--", label="Previous home")
        ax.set_ylabel(f"{'XYZ'[j]} (mm)")
    axs[0].legend(fontsize=8, ncol=3)
    diff = hands - hand_home
    vmax = np.percentile(np.abs(diff), 98)
    im = axs[3].imshow(diff.T, aspect="auto", origin="lower", extent=(ids[0]-.5, ids[-1]+.5, -.5, 12.5),
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axs[3].set(xlabel="Episode index (collection order)", ylabel="Hand motor", yticks=range(13))
    fig.colorbar(im, ax=axs[3], label="Start − home (rad), color clipped at 98%", extend="both")
    fig.suptitle("Variation over collection order: arm starts and hand offsets from configured home")
    save(fig, out, "04_starts_over_collection")

    events = csv_rows(args.run / "events.csv")
    t0 = next(int(r["monotonic_ns"]) for r in events if r["event"] == "observe")
    measured = csv_rows(args.run / "piper.csv")
    commands = csv_rows(args.run / "commands.csv")
    mt = np.array([(int(r["monotonic_ns"]) - t0)/1e9 for r in measured])
    ct = np.array([(int(r["monotonic_ns"]) - t0)/1e9 for r in commands])
    mp = np.array([[float(r[f"fw_{axis}"])*1000 for axis in "xyz"] for r in measured])
    cp = np.array([[float(r[axis])*1000 for axis in "xyz"] for r in commands])
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained", sharex=True)
    for row, paths in enumerate((np.array(trajectories), np.array(deltas))):
        for j in range(3):
            ax = axs[row, j]
            low, median, high = np.nanpercentile(paths[:, :, j], [5, 50, 95], axis=0)
            ax.fill_between(grid, low, high, color=BLUE, alpha=.18, label="Demonstration 5–95%")
            ax.plot(grid, median, color=BLUE, label="Demonstration median")
            offset = 0 if row == 0 else live[j]*1000
            mask = (mt >= 0) & (mt <= 5)
            ax.plot(mt[mask], mp[mask, j]-offset, color=LIVE, label="Measured deployment")
            mask = (ct >= 0) & (ct <= 5)
            ax.plot(ct[mask], cp[mask, j]-offset, color=CURRENT, linestyle="--", label="Deployment command")
            ax.set(xlabel="Seconds from first observation", ylabel=f"{'XYZ'[j]} {'position' if row == 0 else 'displacement'} (mm)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("First 5 seconds: dataset aligned to first synced camera frame; deployment to first observe\nPointwise bands describe timing variation; they are not a target path or task success boundary")
    save(fig, out, "05_early_arm_motion")
    stats["live_xyz_at_seconds_mm"] = {str(t): [float(np.interp(t, mt, mp[:, j])) for j in range(3)] for t in (0, .5, 1, 2, 3, 5)}
    stats["dataset_median_xyz_at_seconds_mm"] = {str(t): np.nanmedian(np.array(trajectories)[:, np.argmin(abs(grid-t))], axis=0).tolist() for t in (0, .5, 1, 2, 3, 5)}

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for j, ax in enumerate(axs.flat):
        ax.hist(joints[valid_ik, j], bins=20, color=BLUE, alpha=.75)
        ax.axvline(qhome[j], color=CURRENT, linewidth=2, label="Current home joints")
        if old_q is not None:
            ax.axvline(old_q[j], color=PREVIOUS, linestyle="--", label="Previous home joints")
        ax.set(xlabel=f"Joint {j+1} (deg)", ylabel="Episodes")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"IK-derived Piper start angles ({valid_ik.sum()}/{len(ids)} valid fits) — not measured dataset joints\nFixed seed at current home; solutions are branch/model dependent")
    save(fig, out, "06_ik_derived_arm_joints")
    (out / "statistics.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: stats[k] for k in ("episodes", "xyz_mm", "home_rotation_distance_to_mean_deg", "ik_valid_count", "ik_max_residual_mm_deg", "live_xyz_at_seconds_mm", "first_sync_pose_error_over_10ms_count")}, indent=2))
    print(f"Saved plots and statistics to {out}")


if __name__ == "__main__":
    main()
