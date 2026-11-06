#!/usr/bin/env python3
"""Read-only audit of calibrated Quest and same-time Vicon flange trajectories.

Thresholds are screening criteria, not a robot safety certification. No trajectories
are filtered, repaired, or written by this program. IK is a separate audit.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_collection import postprocess_dataset as post
from scripts.dataset import postprocess_dataset as base


def stats(values):
    values = np.asarray(values)
    if not len(values):
        return None
    return {"max": float(np.max(values)), "p99": float(np.percentile(values, 99)),
            "median": float(np.median(values)), "mean": float(np.mean(values))}


def runs(mask, times):
    """Durations of contiguous intervals selected by a per-interval mask."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(b), float(times[b] - times[a]))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def compare(qt, qxyz, qr, mt, mxyz, mr, shift):
    shifted = qt + shift
    valid = (shifted >= mt[0]) & (shifted <= mt[-1])
    times = shifted[valid]
    if not len(times):
        return {"samples": 0}
    mpos = np.column_stack([np.interp(times, mt, mxyz[:, i]) for i in range(3)])
    mrot = Slerp(mt, mr)(times)
    distances = np.linalg.norm(qxyz[valid] - mpos, axis=1)
    angles = np.degrees((qr[valid] * mrot.inv()).magnitude())
    bias = np.median(qxyz[valid] - mpos, axis=0)
    over = distances > .20
    edges = np.diff(np.r_[False, over, False].astype(int))
    indices = np.flatnonzero(valid)
    intervals = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        intervals.append(dict(start_sample=int(indices[a]), end_sample=int(indices[b-1]),
                              start_quest_elapsed_s=float(qt[indices[a]]),
                              end_quest_elapsed_s=float(qt[indices[b-1]]),
                              duration_s=float(times[b-1]-times[a]), samples=int(b-a),
                              max_deviation_m=float(np.max(distances[a:b]))))
    return {"samples": int(valid.sum()), "quest_timestamp_shift_s": shift,
            "position_deviation_m": stats(distances),
            "position_deviations_over_0_20_m": int(np.sum(distances > .20)),
            "large_deviation_intervals": intervals,
            "first_overlap_position_deviation_m": float(distances[0]),
            "orientation_deviation_deg": stats(angles),
            "median_position_bias_m": bias.tolist(),
            "bias_removed_position_deviation_m": stats(np.linalg.norm(qxyz[valid] - mpos - bias, axis=1))}


def audit_episode(path, calibration):
    ep = base.RawEpisode(path.name, int(path.name.split("_")[-1]), path,
                         path / "camera.mp4", path / "camera_timestamps.csv",
                         path / "leader_hand.csv", path / "piper.csv", path / "metadata.json")
    rows = base.read_csv_rows(path / "quest.csv")
    raw = np.array([[float(row[key]) for key in ("px", "py", "pz", "qx", "qy", "qz", "qw")] for row in rows])
    seq = np.array([int(row["seq"]) for row in rows])
    _, qeef, qns, _ = post.convert_quest(ep, calibration, absolute=True)
    with contextlib.redirect_stdout(io.StringIO()):
        _, meef, mns, _, _ = post.convert_mocap(ep, calibration, None, absolute=True)
    origin = qns[0]
    qt = (qns - origin) / 1e9
    mt = (mns - origin) / 1e9
    dt = np.diff(qt)
    qxyz = qeef[:, :3].astype(float)
    qr = Rotation.from_rotvec(qeef[:, 3:])
    mr = Rotation.from_rotvec(meef[:, 3:])
    raw_steps = np.linalg.norm(np.diff(raw[:, :3], axis=0), axis=1)
    steps = np.linalg.norm(np.diff(qxyz, axis=0), axis=1)
    angular_steps = np.degrees((qr[1:] * qr[:-1].inv()).magnitude())
    repeated = runs(np.all(raw[1:] == raw[:-1], axis=1), qt)
    long_repeated = [dict(start_s=float(qt[a]), end_s=float(qt[b]), duration_s=d)
                     for a, b, d in repeated if d > .5]
    # Position holds only count as suspicious when the same-time reference moved.
    stale_windows = []
    shift = -float(calibration.get("latency_s", 0.))
    for a in range(len(qt)):
        b = int(np.searchsorted(qt, qt[a] + .5))
        if b >= len(qt) or qt[a] + shift < mt[0] or qt[b] + shift > mt[-1]:
            continue
        qmotion = float(np.max(np.linalg.norm(qxyz[a:b+1] - qxyz[a], axis=1)))
        if qmotion > .005:
            continue
        mpos = np.column_stack([np.interp(qt[a:b+1] + shift, mt, meef[:, i]) for i in range(3)])
        mmotion = float(np.max(np.linalg.norm(mpos - mpos[0], axis=1)))
        if mmotion > .02:
            stale_windows.append(dict(start_sample=a, end_sample=b,
                                      start_s=float(qt[a]), end_s=float(qt[b]),
                                      quest_motion_m=qmotion, mocap_motion_m=mmotion))
    jump_events = []
    for i in np.flatnonzero(steps > .20):
        end = int(i + 1)
        before = []
        for window in (.5, 1., 2.):
            a = int(np.searchsorted(qt, qt[end] - window))
            times = qt[a:end] + shift
            valid = (times >= mt[0]) & (times <= mt[-1])
            times = times[valid]
            positions = qxyz[a:end][valid]
            if not len(times):
                continue
            mocap = np.column_stack([np.interp(times, mt, meef[:, j]) for j in range(3)])
            before.append(dict(window_s=window, start_sample=a, end_sample=end-1,
                               quest_excursion_m=float(np.max(np.linalg.norm(positions-positions[0], axis=1))),
                               mocap_excursion_m=float(np.max(np.linalg.norm(mocap-mocap[0], axis=1)))))
        jump_events.append(dict(sample_index=end, elapsed_s=float(qt[end]),
                                step_m=float(steps[i]), gap_s=float(dt[i]), before_jump=before))
    return {"episode": path.name, "quest_samples": len(rows), "mocap_samples": len(mt),
            "duration_s": float(qt[-1]), "quest_sample_gaps_s": stats(dt),
            "quest_gaps_over_0_5_s": int(np.sum(dt > .5)),
            "raw_position_steps_m": stats(raw_steps),
            "flange_position_steps_m": stats(steps),
            "flange_steps_over_0_20_m": int(np.sum(steps > .20)),
            "large_step_events": jump_events,
            "raw_steps_over_0_20_m": int(np.sum(raw_steps > .20)),
            "flange_speed_m_s": stats(steps / dt),
            "orientation_steps_deg": stats(angular_steps),
            "sequence_nonincreasing": int(np.sum(np.diff(seq) <= 0)),
            "longest_identical_pose_s": max([d for _, _, d in repeated], default=0.),
            "identical_pose_holds_over_0_5_s": long_repeated,
            "suspicious_stale_windows": stale_windows,
            "same_recorded_timestamp": compare(qt, qxyz, qr, mt, meef[:, :3], mr, 0.),
            "calibration_latency_corrected": compare(qt, qxyz, qr, mt, meef[:, :3], mr,
                                                       -float(calibration.get("latency_s", 0.)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    post.validate_calibration(calibration)
    episodes = []
    errors = []
    for path in sorted((args.dataset / "raw").glob("episode_*")):
        try:
            episodes.append(audit_episode(path, calibration))
        except Exception as exc:
            errors.append({"episode": path.name, "error": str(exc)})
        if (len(episodes) + len(errors)) % 20 == 0:
            print(f"Audited {len(episodes)} episodes; {len(errors)} errors", flush=True)
    report = {"dataset": str(args.dataset.resolve()), "calibration": str(args.calibration.resolve()),
              "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
              "thresholds": {"large_jump_m": .20, "large_deviation_m": .20, "long_gap_s": .5,
                             "long_identical_pose_s": .5, "stale_window_s": .5,
                             "stale_max_quest_motion_m": .005, "stale_min_mocap_motion_m": .02},
              "notes": ["Mocap uses the postprocessor's Vicon frame clock and default 20 ms smoothing.",
                        "Comparisons interpolate mocap only over clock overlap; no extrapolation.",
                        "Quest timestamp shift is minus calibration latency_s, following its latency_note.",
                        "No poses are filtered or repaired. This audit does not test robot IK or collision safety."],
              "episodes": episodes, "errors": errors}
    failures = []
    for episode in episodes:
        reasons = []
        if episode["flange_steps_over_0_20_m"]:
            reasons.append("flange_position_jump_over_20_cm")
        if episode["calibration_latency_corrected"]["position_deviations_over_0_20_m"]:
            reasons.append("same_time_mocap_deviation_over_20_cm")
        if episode["quest_gaps_over_0_5_s"]:
            reasons.append("quest_packet_gap_over_0_5_s")
        if episode["identical_pose_holds_over_0_5_s"]:
            reasons.append("identical_pose_hold_over_0_5_s")
        if episode["suspicious_stale_windows"]:
            reasons.append("quest_stationary_while_mocap_moves")
        if reasons:
            failures.append(dict(episode=episode["episode"], reasons=reasons))
    report["failed_episodes"] = failures
    report["summary"] = {
        "episodes_audited": len(episodes), "episode_errors": len(errors),
        "failed_episode_count": len(failures),
        "large_jump_count": sum(e["flange_steps_over_0_20_m"] for e in episodes),
        "suspicious_stale_window_count": sum(len(e["suspicious_stale_windows"]) for e in episodes),
        "strict_1mm_stale_window_count": sum(sum(w["quest_motion_m"] <= .001 for w in e["suspicious_stale_windows"]) for e in episodes),
        "suspicious_stale_episode_count": sum(bool(e["suspicious_stale_windows"]) for e in episodes),
        "large_deviation_sample_count": sum(e["calibration_latency_corrected"]["position_deviations_over_0_20_m"] for e in episodes),
        "maximum_flange_step_m": max(e["flange_position_steps_m"]["max"] for e in episodes),
        "maximum_packet_gap_s": max(e["quest_sample_gaps_s"]["max"] for e in episodes),
        "median_episode_mean_deviation_m": float(np.median([e["calibration_latency_corrected"]["position_deviation_m"]["mean"] for e in episodes])),
        "median_episode_position_bias_m": np.median([e["calibration_latency_corrected"]["median_position_bias_m"] for e in episodes], axis=0).tolist(),
        "starts_deviating_over_20_cm": [e["episode"] for e in episodes if e["calibration_latency_corrected"]["first_overlap_position_deviation_m"] > .20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}: {len(episodes)} episodes, {len(errors)} errors", flush=True)


if __name__ == "__main__":
    main()
