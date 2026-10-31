#!/usr/bin/env python3
"""Fit and validate a Quest -> Piper calibration from quest_calibration_sequence.py runs.

    python scripts/calib/fit_quest_calibration.py output/quest_calibration/<run>
    python scripts/calib/fit_quest_calibration.py RUN --holdout rect_xy circle_xz
    python scripts/calib/fit_quest_calibration.py MOCAP_RUN --tracker mocap   # Vicon reference run

Steps:
  1. Piper reference = joint-feedback FK (fk_* position, fk_roll/pitch/yaw as
     R = Rz(yaw) Ry(pitch) Rx(roll)). Samples are labelled with the segment that
     was being commanded; homing and warmup are excluded.
  2. Quest-to-Piper latency is found by grid search (affine position fit) and
     Quest is resampled onto Piper timestamps at that shift (linear positions,
     SLERP orientations; gaps over 50 ms are dropped).
  3. Candidate models are fitted on the training segments with
     get_coord_transform.fit_pose_transform, each segment as its own recording
     (own origin, so slow Quest drift between segments is not fitted):
     rigid / similarity (each with and without reflection) / linear, each with
     and without the rotating controller-to-flange offset (--fit-offset).
  4. Every model is scored on the held-out segments exactly as teleoperation
     uses it: transform_chunk anchored at the segment's first Piper pose.
     Position error is reported anchored and centered (anchor noise removed);
     orientation error is the angle between mapped and FK orientations.
  5. The model with the lowest held-out anchored RMSE is refitted on all
     segments, scored by leave-one-segment-out against the rigid baseline, and
     written to quest_calibration.json (same schema as get_coord_transform.py,
     plus "latency_s"). A report figure and model_comparison.json are saved.

Apply the calibration with get_coord_transform.transform_chunk on Quest poses
shifted by latency_s (Quest sample at t + latency_s matches Piper at t).
Calibration is only valid for the Quest tracking session it was recorded in:
recentering the Quest or moving its tracking origin changes the frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.calib.get_coord_transform import fit_pose_transform, transform_chunk  # noqa: E402

EXCLUDED_LABELS = ("home", "warmup")
DEFAULT_HOLDOUT = ("rect_xy", "circle_xz")
MAX_GAP_S = 0.05
MODELS = [  # name, model, allow_reflection, fit_offset
    ("rigid", "rigid", False, False),
    ("rigid+refl", "rigid", True, False),
    ("similarity", "similarity", False, False),
    ("similarity+refl", "similarity", True, False),
    ("linear", "linear", False, False),
    ("rigid+offset", "rigid", False, True),
    ("rigid+refl+offset", "rigid", True, True),
    ("similarity+offset", "similarity", False, True),
    ("similarity+refl+offset", "similarity", True, True),
    ("linear+offset", "linear", False, True),
]
COLORS = {"fk": "#2a78d6", "baseline": "#8a8984", "best": "#1baf7a"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


TRACKERS = {
    # name: (file, position columns, quaternion columns, meters per unit, recording flag)
    "quest": ("quest.csv", ("px", "py", "pz"), ("qx", "qy", "qz", "qw"), 1.0, "--quest-port"),
    "mocap": ("mocap.csv", ("x", "y", "z"), ("qx", "qy", "qz", "qw"), 0.001, "Vicon (omit --no-mocap)"),
}


def load_run(run_dir, tracker="quest"):
    piper = read_csv(run_dir / "piper.csv")
    commands = read_csv(run_dir / "commands.csv")
    filename, position_fields, quat_fields, scale, flag = TRACKERS[tracker]
    tracker_path = run_dir / filename
    if not tracker_path.is_file():
        raise FileNotFoundError(f"{tracker_path} missing; record the sequence with {flag}")
    quest = read_csv(tracker_path)
    if "valid" in quest[0]:
        quest = [row for row in quest if row["valid"] == "1"]  # Vicon lost-object frames
    t = np.array([int(row["monotonic_ns"]) for row in piper]) / 1e9
    position = np.array([[float(row[f"fk_{axis}"]) for axis in "xyz"] for row in piper])
    euler = np.array([[float(row[f"fk_{name}"]) for name in ("roll", "pitch", "yaw")] for row in piper])
    quat = Rotation.from_euler("xyz", euler).as_quat()
    t_cmd = np.array([int(row["monotonic_ns"]) for row in commands]) / 1e9
    index = np.searchsorted(t_cmd, t, side="right") - 1
    labels = np.array([commands[i]["label"] if 0 <= i < len(commands) else "home" for i in index], dtype=object)
    labels[(t < t_cmd[0]) | (t > t_cmd[-1])] = "home"
    tq = np.array([int(row["monotonic_ns"]) for row in quest]) / 1e9
    tq, unique = np.unique(tq, return_index=True)
    q_position = scale * np.array([[float(row[name]) for name in position_fields] for row in quest])[unique]
    q_quat = np.array([[float(row[name]) for name in quat_fields] for row in quest])[unique]
    valid = np.linalg.norm(q_quat, axis=1) > 0.5
    return dict(t=t, position=position, quat=quat, labels=labels, tq=tq[valid],
                q_position=q_position[valid], q_quat=q_quat[valid] / np.linalg.norm(q_quat[valid], axis=1)[:, None])


def resample_positions(t_src, values, t_query):
    i1 = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    i0 = i1 - 1
    gap = t_src[i1] - t_src[i0]
    ok = (t_query >= t_src[0]) & (t_query <= t_src[-1]) & (gap <= MAX_GAP_S)
    weight = np.where(gap > 0, (t_query - t_src[i0]) / np.where(gap > 0, gap, 1), 0)
    return values[i0] + weight[:, None] * (values[i1] - values[i0]), ok


def estimate_latency(run, mask, max_lag_s):
    """Shift minimizing an affine position fit; coarse 10 ms grid, then 1 ms refinement."""
    def cost(lag):
        q, ok = resample_positions(run["tq"], run["q_position"], run["t"] + lag)
        ok &= mask
        if ok.sum() < 100:
            return math.inf
        design = np.column_stack([q[ok], np.ones(ok.sum())])
        solution = np.linalg.lstsq(design, run["position"][ok], rcond=None)[0]
        return float(np.sqrt(np.mean(np.sum((design @ solution - run["position"][ok]) ** 2, axis=1))))
    coarse = np.arange(-max_lag_s, max_lag_s + 1e-9, 0.01)
    best = coarse[np.argmin([cost(lag) for lag in coarse])]
    fine = np.arange(best - 0.01, best + 0.0105, 0.001)
    costs = [cost(lag) for lag in fine]
    return float(fine[int(np.argmin(costs))]), float(min(costs))


def synchronized(run, lag, step):
    q, ok = resample_positions(run["tq"], run["q_position"], run["t"] + lag)
    inside = (run["t"] + lag >= run["tq"][0]) & (run["t"] + lag <= run["tq"][-1])
    rotations = Rotation.from_quat(run["q_quat"])
    q_quat = np.full((len(run["t"]), 4), np.nan)
    q_quat[inside] = Slerp(run["tq"], rotations)(run["t"][inside] + lag).as_quat()
    keep = ok & inside & ~np.isin(run["labels"], EXCLUDED_LABELS)
    keep &= np.arange(len(keep)) % step == 0
    return dict(label=run["labels"][keep], quest=q[keep], quest_quat=q_quat[keep],
                piper=run["position"][keep], piper_quat=run["quat"][keep])


def split_segments(data):
    """Contiguous blocks of one label -> list of (label, slice)."""
    labels = data["label"]
    blocks, start = [], 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            blocks.append((labels[start], slice(start, index)))
            start = index
    return blocks


def concat(datasets, blocks):
    parts = [(data, block) for data, (label, block) in blocks]
    arrays = {key: np.concatenate([data[key][block] for data, block in parts])
              for key in ("quest", "quest_quat", "piper", "piper_quat")}
    lengths = [block.stop - block.start for _data, block in parts]
    return arrays, lengths


def score(calibration, data, block):
    quest, quest_quat = data["quest"][block], data["quest_quat"][block]
    piper, piper_quat = data["piper"][block], data["piper_quat"][block]
    targets, target_quat = transform_chunk(quest, quest_quat, piper[0], piper_quat[0], calibration)
    error = 1000 * (targets - piper)
    centered = error - error.mean(axis=0)
    angle = (Rotation.from_quat(target_quat) * Rotation.from_quat(piper_quat).inv()).magnitude()
    return {"anchored_rmse_mm": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
            "centered_rmse_mm": float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))),
            "max_mm": float(np.linalg.norm(error, axis=1).max()),
            "orientation_rmse_deg": float(np.degrees(np.sqrt(np.mean(angle ** 2)))),
            "samples": int(len(error))}, targets


def fit(arrays, lengths, model, reflection, offset):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_pose_transform(arrays["quest"], arrays["piper"], arrays["quest_quat"], arrays["piper_quat"],
                                  recording_lengths=lengths, model=model, allow_reflection=reflection,
                                  fit_offset=offset)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--tracker", choices=sorted(TRACKERS), default="quest",
                        help="Which recorded tracker to calibrate against Piper FK (mocap = Vicon reference run).")
    parser.add_argument("--holdout", nargs="+", default=list(DEFAULT_HOLDOUT),
                        help="Segment labels kept out of model selection fitting.")
    parser.add_argument("--output-dir", type=Path, help="Default: <first run>/calibration")
    parser.add_argument("--max-lag-s", type=float, default=0.3)
    parser.add_argument("--fit-rate-hz", type=float, default=50.0, help="Decimate Piper-rate samples to this rate.")
    parser.add_argument("--models", nargs="+", choices=[name for name, *_ in MODELS],
                        default=[name for name, *_ in MODELS])
    parser.add_argument("--apply", type=Path, metavar="CALIBRATION_JSON",
                        help="Do not select a model: score this existing calibration on the given run(s), "
                             "next to fresh fits of the same model on the new data.")
    args = parser.parse_args(argv)
    if args.apply is not None:
        return apply_existing(args)

    runs, datasets, latencies = [], [], []
    for run_dir in args.run_dirs:
        run = load_run(run_dir, args.tracker)
        moving = ~np.isin(run["labels"], EXCLUDED_LABELS)
        lag, lag_cost = estimate_latency(run, moving, args.max_lag_s)
        if abs(abs(lag) - args.max_lag_s) < 0.002:
            print(f"WARNING: {run_dir.name}: latency at the search limit; increase --max-lag-s")
        rate = 1.0 / np.median(np.diff(run["t"]))
        data = synchronized(run, lag, max(1, round(rate / args.fit_rate_hz)))
        runs.append(run_dir)
        datasets.append(data)
        latencies.append(lag)
        print(f"{run_dir.name}: {args.tracker} latency {1000 * lag:+.0f} ms (affine fit RMSE {1000 * lag_cost:.2f} mm), "
              f"{len(data['label'])} synchronized samples")

    blocks = [(data, block) for data in datasets for block in split_segments(data)]
    labels = sorted({label for _data, (label, _block) in blocks})
    holdout = [label for label in args.holdout if label in labels]
    if len(holdout) < len(args.holdout):
        print(f"WARNING: holdout segments not found: {sorted(set(args.holdout) - set(holdout))}")
    if not holdout:
        raise ValueError(f"No holdout segment present; choose from {labels}")
    train_blocks = [entry for entry in blocks if entry[1][0] not in holdout]
    test_blocks = [entry for entry in blocks if entry[1][0] in holdout]
    print(f"Training segments: {sorted({entry[1][0] for entry in train_blocks})}")
    print(f"Held-out segments: {holdout}")
    arrays, lengths = concat(datasets, train_blocks)

    comparison = {}
    print(f"\n{'model':<24} {'train anch':>10} {'held-out anchored':>18} {'centered':>9} {'orient':>7}  det")
    for name, model, reflection, offset in MODELS:
        if name not in args.models:
            continue
        try:
            calibration = fit(arrays, lengths, model, reflection, offset)
        except (ValueError, np.linalg.LinAlgError) as exc:
            comparison[name] = {"error": str(exc)}
            print(f"{name:<24} failed: {exc}")
            continue
        train = [score(calibration, data, block)[0] for data, (_label, block) in train_blocks]
        test = {f"{label}": score(calibration, data, block)[0] for data, (label, block) in test_blocks}
        mean = lambda rows, key: float(np.mean([row[key] for row in rows]))
        comparison[name] = {
            "train_anchored_rmse_mm": mean(train, "anchored_rmse_mm"),
            "heldout_anchored_rmse_mm": mean(test.values(), "anchored_rmse_mm"),
            "heldout_centered_rmse_mm": mean(test.values(), "centered_rmse_mm"),
            "heldout_orientation_rmse_deg": mean(test.values(), "orientation_rmse_deg"),
            "heldout": test, "determinant": calibration["determinant"],
            "tracked_point_offset_m": calibration.get("tracked_point_offset", {}).get("vector"),
        }
        info = comparison[name]
        print(f"{name:<24} {info['train_anchored_rmse_mm']:9.2f}  {info['heldout_anchored_rmse_mm']:17.2f} "
              f"{info['heldout_centered_rmse_mm']:9.2f} {info['heldout_orientation_rmse_deg']:6.2f}°  "
              f"{info['determinant']:+.3f}")

    scored = {name: info for name, info in comparison.items() if "error" not in info}
    if "rigid" not in scored:
        raise ValueError("Rigid baseline failed; the recording lacks 3D motion")
    best = min(scored, key=lambda name: scored[name]["heldout_anchored_rmse_mm"])
    _, best_model, best_reflection, best_offset = next(entry for entry in MODELS if entry[0] == best)
    print(f"\nBest on held-out segments: {best} "
          f"({scored[best]['heldout_anchored_rmse_mm']:.2f} mm vs rigid {scored['rigid']['heldout_anchored_rmse_mm']:.2f} mm)")

    # Leave-one-segment-out for the rigid baseline and the selected model.
    loso = {"rigid": {}, best: {}}
    for name, model, reflection, offset in (("rigid", "rigid", False, False),
                                            (best, best_model, best_reflection, best_offset)):
        for index, (data, (label, block)) in enumerate(blocks):
            others = [entry for other, entry in enumerate(blocks) if other != index]
            try:
                calibration = fit(*concat(datasets, others), model, reflection, offset)
                loso[name][label] = score(calibration, data, block)[0]
            except (ValueError, np.linalg.LinAlgError) as exc:
                loso[name][label] = {"error": str(exc)}
    print(f"\nLeave-one-segment-out anchored RMSE (mm): {'segment':<12} {'rigid':>7} {best:>12}")
    for label in [label for _data, (label, _block) in blocks]:
        row = [loso[name][label].get("anchored_rmse_mm", float("nan")) for name in ("rigid", best)]
        print(f"{'':42}{label:<12} {row[0]:7.2f} {row[1]:12.2f}")

    final = fit(*concat(datasets, blocks), best_model, best_reflection, best_offset)
    final["latency_s"] = float(np.mean(latencies))
    final["latency_note"] = "Quest sample at t + latency_s corresponds to Piper FK at t; shift Quest timestamps before mapping."
    final["calibration_runs"] = [str(path) for path in runs]
    final["selected_model"] = best
    final["piper_reference"] = "joint-feedback FK (pyAgxArm robot.fk), meters"
    final["validation"] = {"heldout_segments": holdout, "heldout": scored[best]["heldout"],
                           "rigid_heldout": scored["rigid"]["heldout"],
                           "leave_one_segment_out": loso}
    output_dir = args.output_dir or args.run_dirs[0] / f"calibration_{args.tracker}"
    final["tracker"] = args.tracker
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quest_calibration.json").write_text(json.dumps(final, indent=2, allow_nan=False) + "\n")
    (output_dir / "model_comparison.json").write_text(json.dumps(
        {"latencies_s": latencies, "holdout": holdout, "models": comparison, "leave_one_segment_out": loso},
        indent=2) + "\n")
    report(output_dir / "report.png", comparison, loso, best, test_blocks, datasets, arrays, lengths,
           best_model, best_reflection, best_offset)
    print(f"\nWrote {output_dir / 'quest_calibration.json'}, model_comparison.json and report.png")
    return 0


def apply_existing(args):
    """Score a saved calibration on new runs without refitting it.

    Comparisons on the same data, same model type:
      fresh LOSO  - refit on the new run's other segments (fair: never sees the segment)
      fresh all   - refit on every new segment (optimistic best case)
    The saved calibration's latency_s is used for resampling; the new run's own
    latency estimate is printed for reference.
    """
    calibration = json.loads(args.apply.read_text())
    name = calibration.get("selected_model")
    entry = next((item for item in MODELS if item[0] == name), None)
    if entry is None:
        raise ValueError(f"{args.apply} has no recognised selected_model ({name!r})")
    _, model, reflection, offset = entry
    lag = float(calibration.get("latency_s", 0.0))
    tracker = calibration.get("tracker", args.tracker)
    print(f"Applying {args.apply} ({name}, latency {1000 * lag:+.0f} ms, tracker {tracker})")

    datasets = []
    for run_dir in args.run_dirs:
        run = load_run(run_dir, tracker)
        own_lag, _ = estimate_latency(run, ~np.isin(run["labels"], EXCLUDED_LABELS), args.max_lag_s)
        rate = 1.0 / np.median(np.diff(run["t"]))
        datasets.append(synchronized(run, lag, max(1, round(rate / args.fit_rate_hz))))
        print(f"{run_dir.name}: own latency estimate {1000 * own_lag:+.0f} ms (using the saved {1000 * lag:+.0f} ms)")
    blocks = [(data, block) for data in datasets for block in split_segments(data)]
    fresh_all = fit(*concat(datasets, blocks), model, reflection, offset)

    rows = {}
    print(f"\n{'segment':<12} {'saved calib':>11} {'fresh LOSO':>10} {'fresh all':>9}   (anchored position RMSE, mm)  "
          f"orient saved/fresh-all (deg)")
    for index, (data, (label, block)) in enumerate(blocks):
        saved = score(calibration, data, block)[0]
        others = [item for other, item in enumerate(blocks) if other != index]
        try:
            loso = score(fit(*concat(datasets, others), model, reflection, offset), data, block)[0]
        except (ValueError, np.linalg.LinAlgError) as exc:
            loso = {"error": str(exc), "anchored_rmse_mm": float("nan")}
        best = score(fresh_all, data, block)[0]
        rows[label] = {"saved": saved, "fresh_loso": loso, "fresh_all": best}
        print(f"{label:<12} {saved['anchored_rmse_mm']:11.2f} {loso['anchored_rmse_mm']:10.2f} "
              f"{best['anchored_rmse_mm']:9.2f}   {'':30} {saved['orientation_rmse_deg']:.2f} / "
              f"{best['orientation_rmse_deg']:.2f}")
    mean = lambda key: float(np.nanmean([row[key]["anchored_rmse_mm"] for row in rows.values()]))
    summary = {key: mean(key) for key in ("saved", "fresh_loso", "fresh_all")}
    print(f"{'mean':<12} {summary['saved']:11.2f} {summary['fresh_loso']:10.2f} {summary['fresh_all']:9.2f}")

    def parameters(cal):
        return {"linear_matrix": cal["linear_matrix"], "determinant": cal["determinant"],
                "tracked_point_offset_m": cal.get("tracked_point_offset", {}).get("vector")}
    old, new = parameters(calibration), parameters(fresh_all)
    rotation_change = math.degrees(Rotation.from_matrix(
        np.asarray(old["linear_matrix"]) @ np.linalg.inv(np.asarray(new["linear_matrix"]))
        * np.sign(old["determinant"] * new["determinant"])).magnitude()) if model == "rigid" else None
    if old["tracked_point_offset_m"] and new["tracked_point_offset_m"]:
        delta = 1000 * np.linalg.norm(np.subtract(old["tracked_point_offset_m"], new["tracked_point_offset_m"]))
        print(f"Controller offset: saved {np.round(old['tracked_point_offset_m'], 4)} m, "
              f"fresh {np.round(new['tracked_point_offset_m'], 4)} m (difference {delta:.1f} mm)")
    if rotation_change is not None:
        print(f"Frame rotation between saved and fresh fit: {rotation_change:.2f} deg")

    output_dir = args.output_dir or args.run_dirs[0] / f"calibration_{tracker}"
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"applied_{args.apply.parent.parent.name}.json"
    out.write_text(json.dumps({"calibration": str(args.apply), "model": name, "latency_s": lag,
                               "mean_anchored_rmse_mm": summary, "segments": rows,
                               "saved_parameters": old, "fresh_parameters": new,
                               "frame_rotation_change_deg": rotation_change}, indent=2) + "\n")
    print(f"Wrote {out}")
    return 0


def report(path, comparison, loso, best, test_blocks, datasets, arrays, lengths, best_model, best_reflection, best_offset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "font.size": 9})
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor="#fcfcfb")
    for axis in axes:
        axis.set_facecolor("#fcfcfb")

    ax = axes[0]
    names = [name for name in comparison if "error" not in comparison[name]]
    values = [comparison[name]["heldout_anchored_rmse_mm"] for name in names]
    colors = [COLORS["best"] if name == best else COLORS["baseline"] for name in names]
    ax.barh(names, values, color=colors)
    for y, value in enumerate(values):
        ax.text(value, y, f" {value:.2f}", va="center", fontsize=8, color="#0b0b0b")
    ax.invert_yaxis()
    ax.set_xlabel("Held-out anchored position RMSE (mm)")
    ax.set_title("Model comparison on held-out segments", loc="left")

    ax = axes[1]
    labels = list(loso["rigid"])
    x = np.arange(len(labels))
    for offset, name, color in ((-0.2, "rigid", COLORS["baseline"]), (0.2, best, COLORS["best"])):
        ax.bar(x + offset, [loso[name][label].get("anchored_rmse_mm", np.nan) for label in labels], 0.38,
               color=color, label=name)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Anchored position RMSE (mm)")
    ax.set_title("Leave-one-segment-out: rigid vs selected", loc="left")
    ax.legend(frameon=False)

    ax = axes[2]
    data, (label, block) = test_blocks[0]
    rigid = fit(arrays, lengths, "rigid", False, False)
    chosen = fit(arrays, lengths, best_model, best_reflection, best_offset)
    piper = 1000 * (data["piper"][block] - data["piper"][block][0])
    _, center_rigid = score(rigid, data, block)
    _, center_best = score(chosen, data, block)
    _, _s, vt = np.linalg.svd(piper - piper.mean(0), full_matrices=False)
    project = lambda points: points @ vt[:2].T
    for points, color, name, style in ((piper, COLORS["fk"], "Piper FK", "-"),
                                       (1000 * (center_rigid - data["piper"][block][0]), COLORS["baseline"], "Quest, rigid", "-"),
                                       (1000 * (center_best - data["piper"][block][0]), COLORS["best"], f"Quest, {best}", "-")):
        xy = project(points)
        ax.plot(xy[:, 0], xy[:, 1], style, color=color, lw=1.2, label=name)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Principal in-plane axis 1 (mm)")
    ax.set_ylabel("Principal in-plane axis 2 (mm)")
    ax.set_title(f"Held-out '{label}' mapped with each calibration", loc="left")
    ax.legend(frameon=False)

    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
