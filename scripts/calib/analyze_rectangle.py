#!/usr/bin/env python3
"""Analyze a rectangle_repeatability.py run: Piper FK vs Vicon (and Quest).

    python scripts/calib/analyze_rectangle.py output/rect_repeatability/<run>

Each external tracker (Vicon, Quest) is resampled onto the Piper log
timestamps and aligned to Piper FK with a rigid (rotation + translation) fit,
after a grid search for the time offset that minimizes the residual. The
rigid fit absorbs the unknown tracker frame and the constant marker/controller
offset from the flange (orientation is held constant during the run); it does
not absorb scale or shape errors, so edge lengths and residuals still expose
kinematic and tracker errors. The rectangle is planar, so a mirrored tracker
frame (e.g. Quest/Unity left-handed axes) is also fitted by a proper rotation;
only the sign of the out-of-plane axis is then arbitrary.

Reported per tracker: fit residual (RMSE/max, per rectangle axis, moving vs
dwell), per-loop RMSE, corner repeatability across loops, and edge lengths
measured from corner means. Piper FK is the joint-feedback FK (fk_* columns);
FK vs commanded tracking is reported as displacement from the start, and the
firmware-reported pose (fw_*) is compared against joint FK.

Every source (FK, Vicon, Quest) is also scored against the commanded path
(summary.json "vs_commanded", vs_commanded.png): trackers keep their latency
from the FK alignment, then each source gets its own rigid fit to the
commanded positions over the settled corner holds, and RMSE is reported
settled, moving, per loop and as per-corner bias. Record Vicon and Quest in
the same run (both mounted on the flange) to compare all three on one motion.

Outputs summary.json, analysis.png and vs_commanded.png in the run directory. Residuals are
training errors of the alignment fit: they combine FK error, tracker noise,
and any residual timing mismatch.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

PLANES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
AXIS_NAMES = "xyz"
COLORS = {"cmd": "#52514e", "fk": "#2a78d6", "mocap": "#eb6834", "quest": "#1baf7a"}
LABELS = {"cmd": "Commanded", "fk": "Piper FK", "mocap": "Vicon", "quest": "Quest"}
TRACKERS = {
    # name: (file, position columns, mm per unit)
    "mocap": ("mocap.csv", ("x", "y", "z"), 1.0),
    "quest": ("quest.csv", ("px", "py", "pz"), 1000.0),
}
MAX_GAP_S = 0.05
DWELL_SETTLE_FRACTION = 0.4


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def column(rows, name, dtype=float):
    return np.array([dtype(row[name]) for row in rows])


def resample(t_src, values, t_query, max_gap_s=MAX_GAP_S):
    """Linear interpolation; NaN outside coverage or across gaps longer than max_gap_s."""
    i1 = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    i0 = i1 - 1
    gap = t_src[i1] - t_src[i0]
    inside = (t_query >= t_src[0]) & (t_query <= t_src[-1]) & (gap <= max_gap_s)
    weight = np.where(gap > 0, (t_query - t_src[i0]) / np.where(gap > 0, gap, 1), 0)
    out = values[i0] + weight[:, None] * (values[i1] - values[i0])
    out[~inside] = np.nan
    return out


def fit_rigid(src, dst):
    """Proper rotation R and translation t minimizing |R src + t - dst|; valid for planar data."""
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    u, _s, vt = np.linalg.svd((src - src_mean).T @ (dst - dst_mean))
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, dst_mean - rotation @ src_mean


def rmse(errors):
    return float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))) if len(errors) else float("nan")


def align_tracker(t_src, src_mm, t_ref, ref_mm, lags_s):
    best = None
    for lag in lags_s:
        sampled = resample(t_src, src_mm, t_ref + lag)
        ok = np.isfinite(sampled).all(axis=1)
        if ok.sum() < 100:
            continue
        rotation, translation = fit_rigid(sampled[ok], ref_mm[ok])
        error = rmse(sampled[ok] @ rotation.T + translation - ref_mm[ok])
        if best is None or error < best[0]:
            best = (error, lag, rotation, translation)
    if best is None:
        raise ValueError("Tracker and Piper logs do not overlap")
    _error, lag, rotation, translation = best
    sampled = resample(t_src, src_mm, t_ref + lag)
    return sampled @ rotation.T + translation, sampled, lag, rotation, translation


def compare_to_commanded(sources, cmd_mm, in_dwell, loop_of, loops, frame, windows, t_fk, corners_mm):
    """Score every source against the commanded path with the same treatment.

    Each source (FK in the Piper frame, trackers in their own frame, already
    shifted by their latency relative to FK) gets its own rigid fit to the
    commanded positions over the settled corner holds only, so the arm lagging
    the moving target does not bias the alignment. Errors are then measured
    over all samples; moving errors include that lag, which is real tracking
    error the arm has whichever sensor observes it.
    """
    results, aligned = {}, {}
    names = [AXIS_NAMES[axis] for axis in frame]
    for name, values in sources.items():
        ok = np.isfinite(values).all(axis=1)
        fit = ok & in_dwell
        fit_on = "dwell"
        if fit.sum() < 100:
            fit, fit_on = ok, "all"
        rotation, translation = fit_rigid(values[fit], cmd_mm[fit])
        moved = values @ rotation.T + translation
        error = moved - cmd_mm
        dwell, moving = ok & in_dwell, ok & ~in_dwell
        corner_errors = {}
        for _loop, corner, w0, w1 in windows:
            mask = (t_fk >= w0) & (t_fk <= w1) & ok
            if mask.any():
                corner_errors.setdefault(corner, []).append(moved[mask].mean(axis=0) - corners_mm[corner])
        corner_bias = {str(corner): float(np.linalg.norm(np.mean(items, axis=0)))
                       for corner, items in sorted(corner_errors.items())}
        results[name] = {
            "fit_on": fit_on, "coverage": float(ok.mean()),
            "fit_rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))),
            "rmse_mm": rmse(error[ok]),
            "dwell_rmse_mm": rmse(error[dwell]),
            "dwell_max_mm": float(np.linalg.norm(error[dwell], axis=1).max()) if dwell.any() else float("nan"),
            "moving_rmse_mm": rmse(error[moving]),
            "moving_max_mm": float(np.linalg.norm(error[moving], axis=1).max()) if moving.any() else float("nan"),
            "dwell_axis_rmse_mm": dict(zip(names, np.sqrt(np.mean(error[dwell][:, frame]**2, axis=0)).tolist())),
            "per_loop_dwell_rmse_mm": [rmse(error[dwell & (loop_of == loop)]) for loop in range(loops)],
            "per_loop_moving_rmse_mm": [rmse(error[moving & (loop_of == loop)]) for loop in range(loops)],
            "corner_bias_mm": corner_bias,
            "corner_errors_mm": {str(corner): np.array(items).tolist() for corner, items in sorted(corner_errors.items())},
        }
        aligned[name] = moved
    return results, aligned


def dwell_windows(commands, t_cmd):
    """[(loop, corner, t_start, t_end)] for settled corner dwells, excluding the initial dwell."""
    windows = []
    start = 0
    for index in range(1, len(commands) + 1):
        if index < len(commands) and all(commands[index][key] == commands[start][key]
                                         for key in ("loop", "phase", "corner")):
            continue
        row = commands[start]
        if row["phase"] == "dwell" and start > 0:
            t0, t1 = t_cmd[start], t_cmd[index - 1]
            windows.append((int(row["loop"]), int(row["corner"]), t0 + DWELL_SETTLE_FRACTION * (t1 - t0), t1))
        start = index
    return windows


def corner_stats(visits):
    """visits: {corner: [(loop, xyz_mm)]} -> repeatability per corner and edge lengths."""
    stats = {}
    means = {}
    for corner, items in sorted(visits.items()):
        points = np.array([point for _loop, point in items])
        mean = points.mean(axis=0)
        distance = np.linalg.norm(points - mean, axis=1)
        means[corner] = mean
        stats[str(corner)] = {"visits": len(points), "rms_mm": float(np.sqrt(np.mean(distance**2))),
                              "max_mm": float(distance.max()),
                              "axis_std_mm": points.std(axis=0, ddof=1).tolist() if len(points) > 1 else None}
    edges = [float(np.linalg.norm(means[(k + 1) % 4] - means[k])) for k in range(4) if k in means and (k + 1) % 4 in means]
    return stats, means, edges


def stability(residual, elapsed, t_fk, windows):
    """Split tracker-FK residual at settled corners into fast jitter and slow drift.

    Jitter is the within-hold standard deviation; drift is the linear trend of
    the per-hold mean offset over the run (fitted per corner, then averaged).
    """
    jitter, holds = [], {}
    for _loop, corner, w0, w1 in windows:
        mask = (t_fk >= w0) & (t_fk <= w1) & np.isfinite(residual).all(axis=1)
        if mask.sum() < 10:
            continue
        jitter.append(residual[mask].std(axis=0))
        holds.setdefault(corner, []).append((elapsed[mask].mean(), residual[mask].mean(axis=0)))
    slopes = [np.polyfit([t for t, _ in items], np.array([o for _, o in items]), 1)[0] * 60
              for items in holds.values() if len(items) >= 3]
    ordered = sorted((t, tuple(o)) for items in holds.values() for t, o in items)
    offsets = np.array([o for _t, o in ordered])
    return {
        "hold_jitter_std_mm": np.mean(jitter, axis=0).tolist() if jitter else None,
        "drift_mm_per_min": np.mean(slopes, axis=0).tolist() if slopes else None,
        "hold_offset_std_mm": offsets.std(axis=0).tolist() if len(offsets) else None,
        "hold_offset_first5_mm": offsets[:5].mean(axis=0).tolist() if len(offsets) >= 5 else None,
        "hold_offset_last5_mm": offsets[-5:].mean(axis=0).tolist() if len(offsets) >= 5 else None,
    }


def per_loop_table(summary, series, fk_error, loop_of, corner_means):
    """One row per loop: FK tracking, tracker-FK residual, and rectangle size seen by each source.

    Edges come from that loop's settled corner means; corner 0 closes each loop,
    so edge 0 (corner 0 -> 1) uses the previous loop's closing corner and is
    unavailable for loop 1.
    """
    table = []
    for loop in range(summary["loops"]):
        mask = loop_of == loop
        row = {"loop": loop + 1, "fk_vs_cmd_rmse_mm": rmse(fk_error[mask])}
        for name, values in series.items():
            if name != "fk":
                residual = values[mask] - series["fk"][mask]
                ok = np.isfinite(residual).all(axis=1)
                row[f"{name}_rmse_mm"] = rmse(residual[ok])
                row[f"{name}_max_mm"] = float(np.linalg.norm(residual[ok], axis=1).max()) if ok.any() else float("nan")
                row[f"{name}_coverage"] = float(ok.mean())
            corners = {corner: point for corner, visits in corner_means[name].items()
                       for visit_loop, point in visits if visit_loop == loop}
            previous = {corner: point for corner, visits in corner_means[name].items()
                        for visit_loop, point in visits if visit_loop == loop - 1}
            path = [previous.get(0), corners.get(1), corners.get(2), corners.get(3), corners.get(0)]
            edges = [float(np.linalg.norm(b - a)) if a is not None and b is not None else float("nan")
                     for a, b in zip(path, path[1:])]
            row[f"{name}_edges_mm"] = edges
        table.append(row)
    return table


def analyze(run_dir, max_lag_s, lag_step_s):
    meta = json.loads((run_dir / "meta.json").read_text())
    u, v = PLANES[meta["plane"]]
    n = 3 - u - v
    frame = [u, v, n]

    piper = load_csv(run_dir / "piper.csv")
    commands = load_csv(run_dir / "commands.csv")
    t_fk = column(piper, "monotonic_ns") / 1e9
    fk = 1000 * np.column_stack([column(piper, f"fk_{axis}") for axis in "xyz"])
    fw = 1000 * np.column_stack([column(piper, f"fw_{axis}", lambda value: float(value or "nan")) for axis in "xyz"])
    t_cmd = column(commands, "monotonic_ns") / 1e9
    cmd = 1000 * np.column_stack([column(commands, axis) for axis in "xyz"])
    t0 = t_cmd[0]
    keep = (t_fk >= t_cmd[0]) & (t_fk <= t_cmd[-1])
    t_fk, fk, fw = t_fk[keep], fk[keep], fw[keep]

    # Commands live in the firmware pose frame; compare joint FK to them as displacements
    # from the start so a constant FK/firmware frame offset is not counted as tracking error.
    cmd_on_fk = cmd[np.clip(np.searchsorted(t_cmd, t_fk, side="right") - 1, 0, len(cmd) - 1)]
    cmd_on_fk = cmd_on_fk - 1000 * np.array(meta["start_pose"][:3]) + 1000 * np.array(meta["start_fk"][:3])
    start_mm = 1000 * np.array(meta["start_fk"][:3])
    loop_of = column(commands, "loop", int)[np.clip(np.searchsorted(t_cmd, t_fk, side="right") - 1, 0, len(cmd) - 1)]
    windows = dwell_windows(commands, t_cmd)
    in_dwell = np.zeros(len(t_fk), dtype=bool)
    for _loop, _corner, w0, w1 in windows:
        in_dwell |= (t_fk >= w0) & (t_fk <= w1)

    series = {"fk": fk}
    raw = {"fk": fk}
    summary = {"run": str(run_dir), "plane": meta["plane"], "status": meta.get("status"),
               "loops": int(loop_of.max()) + 1, "frame_axes": [AXIS_NAMES[axis] for axis in frame],
               "units": "mm", "trackers": {}}

    # The commanded target leads the arm, so only settled dwells measure static accuracy.
    fk_error = fk - cmd_on_fk
    summary["fk_vs_commanded"] = {"dwell_rmse_mm": rmse(fk_error[in_dwell]),
                                  "dwell_max_mm": float(np.linalg.norm(fk_error[in_dwell], axis=1).max()),
                                  "moving_rmse_mm": rmse(fk_error[~in_dwell])}
    fw_ok = np.isfinite(fw).all(axis=1)
    if fw_ok.any():
        gap = fw[fw_ok] - fk[fw_ok]
        summary["firmware_pose_vs_joint_fk"] = {
            "mean_offset_mm": gap.mean(axis=0).tolist(),
            "rmse_after_mean_offset_mm": rmse(gap - gap.mean(axis=0)),
            "max_mm": float(np.linalg.norm(gap, axis=1).max())}

    lags = np.arange(-max_lag_s, max_lag_s + lag_step_s / 2, lag_step_s)
    for name, (filename, fields, scale) in TRACKERS.items():
        path = run_dir / filename
        if not path.is_file():
            continue
        rows = load_csv(path)
        if "valid" in rows[0]:
            rows = [row for row in rows if row["valid"] == "1"]
        stamps = column(rows, "monotonic_ns")
        stamps, unique = np.unique(stamps, return_index=True)
        t_src = stamps / 1e9
        src = scale * np.column_stack([column(rows, field) for field in fields])[unique]
        aligned, raw[name], lag, rotation, translation = align_tracker(t_src, src, t_fk, fk, lags)
        ok = np.isfinite(aligned).all(axis=1)
        residual = aligned - fk
        per_loop = [rmse(residual[ok & (loop_of == loop)]) for loop in range(summary["loops"])]
        summary["trackers"][name] = {
            "samples_used": int(ok.sum()), "coverage": float(ok.mean()),
            "lag_s": float(lag), "lag_at_search_edge": bool(abs(abs(lag) - max_lag_s) < lag_step_s / 2),
            "rotation_tracker_to_piper": rotation.tolist(), "translation_mm": translation.tolist(),
            "rmse_mm": rmse(residual[ok]), "max_mm": float(np.linalg.norm(residual[ok], axis=1).max()),
            "axis_rmse_mm": dict(zip(summary["frame_axes"], np.sqrt(np.mean(residual[ok][:, frame]**2, axis=0)).tolist())),
            "dwell_rmse_mm": rmse(residual[ok & in_dwell]), "moving_rmse_mm": rmse(residual[ok & ~in_dwell]),
            "per_loop_rmse_mm": per_loop,
        }
        series[name] = aligned

    for name in summary["trackers"]:
        summary["trackers"][name].update(stability(series[name] - series["fk"], t_fk - t0, t_fk, windows))

    corner_means = {}
    for name, values in series.items():
        visits = {}
        for loop, corner, w0, w1 in windows:
            mask = (t_fk >= w0) & (t_fk <= w1) & np.isfinite(values).all(axis=1)
            if mask.any():
                visits.setdefault(corner, []).append((loop, values[mask].mean(axis=0)))
        stats, means, edges = corner_stats(visits)
        corner_means[name] = visits
        target = summary if name == "fk" else summary["trackers"][name]
        target["fk_corner_repeatability" if name == "fk" else "corner_repeatability"] = stats
        target["fk_edge_lengths_mm" if name == "fk" else "edge_lengths_mm"] = edges
    summary["per_loop"] = per_loop_table(summary, series, fk_error, loop_of, corner_means)
    summary["commanded_edge_lengths_mm"] = [float(np.linalg.norm(1000 * np.subtract(meta["corners"][(k + 1) % 4], meta["corners"][k])))
                                            for k in range(4)]
    corners_mm = 1000 * (np.array(meta["corners"]) - np.array(meta["start_pose"][:3])) + start_mm
    summary["vs_commanded"], on_cmd = compare_to_commanded(raw, cmd_on_fk, in_dwell, loop_of, summary["loops"],
                                                           frame, windows, t_fk, corners_mm)

    plot(run_dir, summary, t_fk - t0, series, cmd_on_fk, start_mm, frame, loop_of, corner_means)
    plot_vs_commanded(run_dir, summary, t_fk - t0, on_cmd, cmd_on_fk, frame, loop_of)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _figure(rows, cols, figsize):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "font.size": 9})
    figure, axes = plt.subplots(rows, cols, figsize=figsize, facecolor="#fcfcfb")
    for axis in axes.flat:
        axis.set_facecolor("#fcfcfb")
    return plt, figure, axes


def plot(run_dir, summary, t, series, cmd, start_mm, frame, loop_of, corner_means):
    names = [AXIS_NAMES[axis] for axis in frame]
    local = {key: (values - start_mm)[:, frame] for key, values in {**series, "cmd": cmd}.items()}
    trackers = [name for name in series if name != "fk"]
    plt, figure, axes = _figure(2, 2, (12, 10))

    ax = axes[0, 0]
    for name in (*trackers, "fk"):
        ax.plot(local[name][:, 0], local[name][:, 1], color=COLORS[name], lw=0.8, alpha=0.6, label=LABELS[name])
    ax.plot(local["cmd"][:, 0], local["cmd"][:, 1], color=COLORS["cmd"], lw=0.8, ls="--", label=LABELS["cmd"])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"Piper {names[0]} from start (mm)")
    ax.set_ylabel(f"Piper {names[1]} from start (mm)")
    ax.set_title(f"All {summary['loops']} loops in the rectangle plane (trackers rigidly aligned to FK)", loc="left")
    ax.legend(frameon=False, loc="center")

    ax = axes[0, 1]
    for name in trackers:
        error = np.linalg.norm(series[name] - series["fk"], axis=1)
        info = summary["trackers"][name]
        ax.plot(t, error, color=COLORS[name], lw=0.8,
                label=f"{LABELS[name]} − FK (RMSE {info['rmse_mm']:.2f} mm, lag {1000 * info['lag_s']:+.0f} ms)")
    fk_error = np.linalg.norm(series["fk"] - cmd, axis=1)
    ax.plot(t, fk_error, color=COLORS["cmd"], lw=0.8, alpha=0.7, label="FK − commanded (target leads while moving)")
    for boundary in t[np.flatnonzero(np.diff(loop_of)) + 1]:
        ax.axvline(boundary, color="#d6d5d0", lw=0.6, zorder=0)
    ax.set_xlabel("Time (s); vertical lines mark loop starts")
    ax.set_ylabel("Position error magnitude (mm)")
    ax.set_title("Error over time", loc="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1))

    ax = axes[1, 0]
    loops = np.arange(1, summary["loops"] + 1)
    for name in trackers:
        ax.plot(loops, summary["trackers"][name]["per_loop_rmse_mm"], color=COLORS[name], lw=2, marker="o", ms=5,
                label=f"{LABELS[name]} − FK")
    ax.plot(loops, [row["fk_vs_cmd_rmse_mm"] for row in summary["per_loop"]], color=COLORS["cmd"], lw=2,
            marker="s", ms=5, label="FK − commanded")
    ax.set_xticks(loops)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Loop (trial)")
    ax.set_ylabel("RMSE (mm)")
    ax.set_title("Per-trial RMSE", loc="left")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 1]
    for name in ("fk", *trackers):
        deviations = []
        for corner, visits in corner_means[name].items():
            points = (np.array([point for _loop, point in visits]))[:, frame[:2]]
            deviations.append(points - points.mean(axis=0))
        if not deviations:
            continue
        deviations = np.concatenate(deviations)
        stats = summary["fk_corner_repeatability"] if name == "fk" else summary["trackers"][name]["corner_repeatability"]
        worst = max(item["rms_mm"] for item in stats.values())
        ax.scatter(deviations[:, 0], deviations[:, 1], s=22, color=COLORS[name], alpha=0.75,
                   edgecolors="#fcfcfb", linewidths=1, label=f"{LABELS[name]} (worst corner RMS {worst:.3f} mm)")
    ax.axhline(0, color="#8a8984", lw=0.6)
    ax.axvline(0, color="#8a8984", lw=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"{names[0]} deviation from corner mean (mm)")
    ax.set_ylabel(f"{names[1]} deviation from corner mean (mm)")
    ax.set_title("Corner repeatability: settled dwell positions, all 4 corners overlaid", loc="left")
    ax.legend(frameon=False, loc="upper right")

    figure.suptitle(f"Rectangle repeatability: {run_dir.name}", x=0.01, ha="left", fontsize=12)
    figure.tight_layout()
    figure.savefig(run_dir / "analysis.png", dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def plot_vs_commanded(run_dir, summary, t, on_cmd, cmd, frame, loop_of):
    results = summary["vs_commanded"]
    names = [AXIS_NAMES[axis] for axis in frame]
    sources = list(results)
    plt, figure, axes = _figure(2, 2, (12, 9))

    ax = axes[0, 0]
    for name in sources:
        ax.plot(t, np.linalg.norm(on_cmd[name] - cmd, axis=1), color=COLORS[name], lw=0.8, alpha=0.8,
                label=f"{LABELS[name]} (dwell RMSE {results[name]['dwell_rmse_mm']:.2f} mm)")
    for boundary in t[np.flatnonzero(np.diff(loop_of)) + 1]:
        ax.axvline(boundary, color="#d6d5d0", lw=0.6, zorder=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (s); vertical lines mark loop starts")
    ax.set_ylabel("Distance from commanded position (mm)")
    ax.set_title("Error vs commanded over time", loc="left")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    width = 0.8 / len(sources)
    for index, name in enumerate(sources):
        values = [results[name]["dwell_rmse_mm"], results[name]["moving_rmse_mm"]]
        bars = ax.bar(np.arange(2) + (index - (len(sources) - 1) / 2) * width, values, width * 0.92,
                      color=COLORS[name], label=LABELS[name])
        ax.bar_label(bars, fmt="%.2f", padding=2, color="#52514e", fontsize=8)
    ax.set_xticks(np.arange(2), ["Settled at corners", "Moving along edges"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("RMSE vs commanded (mm)")
    ax.set_title("RMSE vs commanded", loc="left")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 0]
    loops = np.arange(1, summary["loops"] + 1)
    for name in sources:
        ax.plot(loops, results[name]["per_loop_dwell_rmse_mm"], color=COLORS[name], lw=2, marker="o", ms=5,
                label=LABELS[name])
    ax.set_xticks(loops)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Loop (trial)")
    ax.set_ylabel("Settled RMSE vs commanded (mm)")
    ax.set_title("Per-trial settled RMSE vs commanded", loc="left")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 1]
    for name in sources:
        points = np.concatenate([np.array(items) for items in results[name]["corner_errors_mm"].values()])
        ax.scatter(points[:, frame[0]], points[:, frame[1]], s=22, color=COLORS[name], alpha=0.75,
                   edgecolors="#fcfcfb", linewidths=1, label=LABELS[name])
    ax.axhline(0, color="#8a8984", lw=0.6)
    ax.axvline(0, color="#8a8984", lw=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"{names[0]} error at corner (mm)")
    ax.set_ylabel(f"{names[1]} error at corner (mm)")
    ax.set_title("Settled corner position minus commanded corner (every visit)", loc="left")
    ax.legend(frameon=False, loc="best")

    figure.suptitle(f"{' / '.join(LABELS[name] for name in sources)} vs commanded: {run_dir.name}\n"
                    "Each source rigidly aligned to the commanded path on settled holds", x=0.01, ha="left", fontsize=12)
    figure.tight_layout()
    figure.savefig(run_dir / "vs_commanded.png", dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def print_per_loop(summary):
    trackers = list(summary["trackers"])
    header = f"{'loop':>4} {'FK-cmd':>7}"
    for name in trackers:
        header += f" {LABELS[name] + '-FK':>9} {'max':>6} {'cov%':>5}"
    for name in ("fk", *trackers):
        header += f"  {LABELS[name] + ' edges 0/1/2/3 (mm)':>29}"
    print("Per-trial (RMSE/max in mm):")
    print(header)
    for row in summary["per_loop"]:
        line = f"{row['loop']:>4} {row['fk_vs_cmd_rmse_mm']:7.3f}"
        for name in trackers:
            line += f" {row[f'{name}_rmse_mm']:9.3f} {row[f'{name}_max_mm']:6.2f} {100 * row[f'{name}_coverage']:5.1f}"
        for name in ("fk", *trackers):
            line += "  " + " ".join(f"{edge:6.2f}" for edge in row[f"{name}_edges_mm"]).rjust(29)
        print(line)


def print_summary(summary):
    fk = summary["fk_vs_commanded"]
    print(f"Run: {summary['run']} ({summary['status']}, {summary['loops']} loops, plane {summary['plane']})")
    print(f"FK vs commanded: dwell RMSE {fk['dwell_rmse_mm']:.3f} mm (max {fk['dwell_max_mm']:.3f}), "
          f"moving RMSE {fk['moving_rmse_mm']:.3f} mm")
    if "firmware_pose_vs_joint_fk" in summary:
        info = summary["firmware_pose_vs_joint_fk"]
        print(f"Firmware pose vs joint FK: mean offset {np.linalg.norm(info['mean_offset_mm']):.3f} mm, "
              f"varying part RMSE {info['rmse_after_mean_offset_mm']:.3f} mm")
    worst = max(item["rms_mm"] for item in summary["fk_corner_repeatability"].values())
    print(f"FK corner repeatability (worst corner RMS): {worst:.4f} mm")
    edges = ", ".join(f"{value:.2f}" for value in summary["commanded_edge_lengths_mm"])
    print(f"Edge lengths (mm): commanded [{edges}]  FK [{', '.join(f'{v:.2f}' for v in summary['fk_edge_lengths_mm'])}]")
    for name, info in summary["trackers"].items():
        worst = max(item["rms_mm"] for item in info["corner_repeatability"].values())
        axes = ", ".join(f"{axis} {value:.3f}" for axis, value in info["axis_rmse_mm"].items())
        print(f"{LABELS[name]} - FK: RMSE {info['rmse_mm']:.3f} mm (max {info['max_mm']:.3f}; {axes}); "
              f"dwell {info['dwell_rmse_mm']:.3f}, moving {info['moving_rmse_mm']:.3f}; "
              f"lag {1000 * info['lag_s']:+.0f} ms; coverage {100 * info['coverage']:.1f}%")
        if info["lag_at_search_edge"]:
            print(f"  WARNING: best lag is at the search limit; rerun with a larger --max-lag-s")
        print(f"  {LABELS[name]} corner repeatability (worst corner RMS): {worst:.4f} mm; "
              f"edges [{', '.join(f'{v:.2f}' for v in info['edge_lengths_mm'])}] mm")
        if info.get("hold_jitter_std_mm") is not None:
            fmt = lambda values: "/".join(f"{value:.2f}" for value in values)
            print(f"  hold jitter (std) {fmt(info['hold_jitter_std_mm'])} mm; "
                  f"drift {fmt(info['drift_mm_per_min'])} mm/min; "
                  f"hold offset first5 {fmt(info['hold_offset_first5_mm'])} -> last5 {fmt(info['hold_offset_last5_mm'])} mm "
                  f"({'/'.join(summary['frame_axes'])})")


def print_vs_commanded(summary):
    axes = "/".join(summary["frame_axes"])
    print("Vs commanded (each source rigidly aligned to the commanded path on settled holds; mm):")
    print(f"{'source':>9} {'settled':>8} {'max':>6} {'moving':>8} {'max':>6} {'overall':>8}"
          f"  {'settled ' + axes:>20}  {'corner bias 0/1/2/3':>23} {'cov%':>6}")
    for name, info in summary["vs_commanded"].items():
        per_axis = "/".join(f"{value:.2f}" for value in info["dwell_axis_rmse_mm"].values())
        bias = "/".join(f"{value:.2f}" for value in info["corner_bias_mm"].values())
        print(f"{LABELS[name]:>9} {info['dwell_rmse_mm']:8.3f} {info['dwell_max_mm']:6.2f} "
              f"{info['moving_rmse_mm']:8.3f} {info['moving_max_mm']:6.2f} {info['rmse_mm']:8.3f}"
              f"  {per_axis:>20}  {bias:>23} {100 * info['coverage']:6.1f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-lag-s", type=float, default=0.3)
    parser.add_argument("--lag-step-s", type=float, default=0.005)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_lag_s) or args.max_lag_s < 0 or not math.isfinite(args.lag_step_s) or args.lag_step_s <= 0:
        parser.error("--max-lag-s must be non-negative and --lag-step-s positive")
    summary = analyze(args.run_dir, args.max_lag_s, args.lag_step_s)
    print_per_loop(summary)
    print_summary(summary)
    print_vs_commanded(summary)
    print(f"Wrote {args.run_dir / 'summary.json'}, {args.run_dir / 'analysis.png'} and {args.run_dir / 'vs_commanded.png'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
