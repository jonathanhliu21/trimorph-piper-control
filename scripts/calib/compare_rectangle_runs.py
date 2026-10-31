#!/usr/bin/env python3
"""Compare analyzed rectangle_repeatability runs side by side.

    python scripts/calib/analyze_rectangle.py RUN_A
    python scripts/calib/analyze_rectangle.py RUN_B
    python scripts/calib/compare_rectangle_runs.py RUN_A RUN_B --output output/rect_repeatability/compare.png

Reads each run's summary.json (written by analyze_rectangle.py) and plots, per
tracker: per-trial tracker-FK RMSE, edge lengths (mean and spread over trials)
next to Piper FK and the commanded 20 mm, and corner repeatability vs hold
jitter. Prints a comparison table. Runs should share the same start pose and
rectangle for the comparison to be like-for-like; the table shows each start.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from analyze_rectangle import COLORS, LABELS  # noqa: E402  (same directory)


def load(run_dir):
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"{summary_path} missing; run analyze_rectangle.py {run_dir} first")
    summary = json.loads(summary_path.read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    return summary, meta


def edge_stats(per_loop, key):
    edges = np.array([row[key] for row in per_loop], dtype=float)
    return np.nanmean(edges, axis=0), np.nanstd(edges, axis=0, ddof=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, help="Figure path (default: compare.png beside the first run).")
    args = parser.parse_args(argv)

    entries = []  # (tracker name, run name, summary, meta)
    fk_rows = []
    for run_dir in args.run_dirs:
        summary, meta = load(run_dir)
        fk_rows.extend(summary["per_loop"])
        for name in summary["trackers"]:
            entries.append((name, run_dir.name, summary, meta))
    if not entries:
        raise ValueError("No tracker data in the given runs")

    print(f"{'source':<8} {'run':<24} {'RMSE':>6} {'max':>6} {'lag ms':>7} {'cov%':>5} {'corner':>7} "
          f"{'jitter x/y/z':>17} {'drift mm/min x/y/z':>20}  edges mean (mm)")
    for name, run, summary, meta in entries:
        info = summary["trackers"][name]
        worst = max(item["rms_mm"] for item in info["corner_repeatability"].values())
        mean, _spread = edge_stats(summary["per_loop"], f"{name}_edges_mm")
        print(f"{LABELS[name]:<8} {run:<24} {info['rmse_mm']:6.3f} {info['max_mm']:6.2f} {1000 * info['lag_s']:+7.0f} "
              f"{100 * info['coverage']:5.1f} {worst:7.3f} "
              f"{'/'.join(f'{v:.2f}' for v in info['hold_jitter_std_mm']):>17} "
              f"{'/'.join(f'{v:+.2f}' for v in info['drift_mm_per_min']):>20}  "
              + " ".join(f"{v:5.2f}" for v in mean))
    fk_mean, fk_spread = edge_stats(fk_rows, "fk_edges_mm")
    print(f"{'Piper FK':<8} {'(all runs)':<24} {'':>6} {'':>6} {'':>7} {'':>5} "
          f"{max(max(item['rms_mm'] for item in s['fk_corner_repeatability'].values()) for _, _, s, _ in entries):7.3f} "
          f"{'':>17} {'':>20}  " + " ".join(f"{v:5.2f}" for v in fk_mean))
    for _name, run, _summary, meta in {entry[1]: entry for entry in entries}.values():
        print(f"start pose {run}: {[round(v, 4) for v in meta['start_pose'][:3]]} m, "
              f"joints {[round(float(np.degrees(v)), 1) for v in meta.get('start_joints', [])]} deg")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "font.size": 9})
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor="#fcfcfb")
    for axis in axes:
        axis.set_facecolor("#fcfcfb")

    ax = axes[0]
    for name, run, summary, _meta in entries:
        values = summary["trackers"][name]["per_loop_rmse_mm"]
        ax.plot(np.arange(1, len(values) + 1), values, color=COLORS[name], lw=2, marker="o", ms=5,
                label=f"{LABELS[name]} − FK ({run})")
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(1, 21))
    ax.set_xlabel("Trial (loop)")
    ax.set_ylabel("RMSE vs Piper FK (mm)")
    ax.set_title("Per-trial tracker − FK RMSE", loc="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12))

    ax = axes[1]
    sources = [("fk", fk_mean, fk_spread)] + [
        (name, *edge_stats(summary["per_loop"], f"{name}_edges_mm")) for name, _run, summary, _meta in entries]
    width = 0.8 / len(sources)
    for index, (name, mean, spread) in enumerate(sources):
        positions = np.arange(4) + (index - (len(sources) - 1) / 2) * width
        ax.bar(positions, mean - 15, width * 0.9, bottom=15, color=COLORS[name], label=LABELS[name],
               yerr=spread, error_kw={"ecolor": "#52514e", "elinewidth": 1, "capsize": 2})
    ax.axhline(20, color=COLORS["cmd"], lw=1, ls="--", label="Commanded 20 mm")
    ax.set_ylim(15, 22)
    ax.set_xticks(np.arange(4), [f"edge {k}\n({'x' if k % 2 == 0 else 'y'})" for k in range(4)])
    ax.set_ylabel("Edge length (mm); bar = mean, whisker = SD over trials")
    ax.set_title("Rectangle size seen by each source (axis starts at 15 mm)", loc="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)

    ax = axes[2]
    labels, corner, jitter, colors = [], [], [], []
    first = entries[0][2]
    labels.append("Piper FK")
    corner.append(max(item["rms_mm"] for item in first["fk_corner_repeatability"].values()))
    jitter.append(np.nan)
    colors.append(COLORS["fk"])
    for name, _run, summary, _meta in entries:
        info = summary["trackers"][name]
        labels.append(LABELS[name])
        corner.append(max(item["rms_mm"] for item in info["corner_repeatability"].values()))
        jitter.append(float(np.linalg.norm(info["hold_jitter_std_mm"])))
        colors.append(COLORS[name])
    positions = np.arange(len(labels))
    ax.bar(positions - 0.2, corner, 0.38, color=colors, label="Corner repeatability (worst corner RMS)")
    ax.bar(positions + 0.2, jitter, 0.38, color=colors, alpha=0.45, hatch="//", edgecolor="#fcfcfb",
           label="Jitter within a hold (std, |xyz|)")
    for x, value in zip(positions - 0.2, corner):
        ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", color="#0b0b0b", fontsize=8)
    for x, value in zip(positions + 0.2, jitter):
        if np.isfinite(value):
            ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", color="#0b0b0b", fontsize=8)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("mm")
    ax.set_title("Repeatability across trials vs short-term jitter", loc="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.08))

    figure.suptitle("Rectangle repeatability comparison: " + ", ".join(path.name for path in args.run_dirs),
                    x=0.01, ha="left", fontsize=12)
    figure.tight_layout()
    output = args.output or args.run_dirs[0].parent / "compare.png"
    figure.savefig(output, dpi=150, facecolor=figure.get_facecolor())
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
