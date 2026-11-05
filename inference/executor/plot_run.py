"""Plot an executor run log: position, velocity, and acceleration per Piper joint and flange axis.

    python inference/executor/plot_run.py output/executor_runs/20261104_120000
    python inference/executor/plot_run.py RUN --start 10 --end 16   # zoom, seconds from the first observation

Measured: the 200 Hz joint feedback in piper.csv, one sample per SDK update,
resampled on a uniform 200 Hz grid with Savitzky-Golay derivatives; the flange
is FK of those joints. Commanded: the EMA-filtered targets in commands.csv (one
per action, e.g. 15 Hz), drawn as held steps; joint targets are IK of each
target seeded at the measured joints. Gray spans: waiting for the predictor
(observe -> chunk received; with --rtc the arm keeps moving during them);
dotted lines: chunk starts; red lines: pedal resets.

Saves joints.png, eef.png, eef_position_error.png, and spectrum.png in the run folder.
The error plot compares measured flange XYZ with each requested target at its
arrival time (one action period later for interpolated joint commands), showing
signed measured-minus-commanded axes and the Euclidean position error in mm.
Measured positions are FK of joint feedback, as in eef.png. Prints, per
joint and flange axis, the velocity RMS in frequency bands for measured vs
commanded motion: measured >> commanded means the arm adds the shake; equal
means it is in the commands (policy or chunk boundaries).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter, welch
from scipy.spatial.transform import Rotation

CONTROL_ROOT = Path(__file__).resolve().parents[2]
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from data_collection.replay_actions import piper_model  # noqa: E402

GRID_HZ = 200.0
# Savitzky-Golay (cubic) windows at 200 Hz. Feedback timing jitter dominates the
# derivatives above ~15 Hz, so plots pass up to ~19 Hz (velocity) and ~15 Hz
# (acceleration); the spectrum and band table use the unsmoothed derivative.
VELOCITY_WINDOW = 13  # 65 ms
ACCELERATION_WINDOW = 17  # 85 ms
BANDS_HZ = ((0.0, 2.0), (2.0, 6.0), (6.0, 12.0), (12.0, 20.0))
PSD_SEGMENT = 400  # 2 s Welch segments (0.5 Hz bins); shorter policy segments are skipped
EULER = ("x", "y", "z", "roll", "pitch", "yaw")
EEF_LABELS = ("x (mm)", "y (mm)", "z (mm)", "roll (deg)", "pitch (deg)", "yaw (deg)")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def column(rows, name):
    return np.array([float(row[name]) if row.get(name) not in (None, "") else np.nan for row in rows])


def ik_path(fk, bounds, poses, seeds):
    """Joint targets for XYZ/Euler poses, each seeded at the measured joints at that time."""
    out = []
    for pose, seed in zip(poses, seeds):
        rotation = Rotation.from_euler("xyz", pose[3:])

        def residual(q):
            p = fk(q)
            return np.r_[p[:3] - pose[:3], .2 * (Rotation.from_euler("xyz", p[3:]) * rotation.inv()).as_rotvec()]

        out.append(least_squares(residual, np.clip(seed, bounds[:, 0], bounds[:, 1]),
                                 bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=100).x)
    return np.array(out)


def load(run):
    run = Path(run)
    piper = read_csv(run / "piper.csv")
    commands = read_csv(run / "commands.csv")
    events = read_csv(run / "events.csv")
    if not piper or not commands:
        raise SystemExit(f"{run}: piper.csv and commands.csv must both have rows")

    # One sample per SDK joint update, timed by its receive stamp on the monotonic clock.
    poll_s = column(piper, "monotonic_ns") / 1e9
    stamp = column(piper, "joint_timestamp_s")
    joints = np.column_stack([column(piper, f"joint_{i}") for i in range(1, 7)])
    fresh = np.r_[True, np.diff(stamp) != 0] & np.isfinite(joints).all(axis=1)
    offset = np.min(poll_s[fresh] - stamp[fresh])  # poll happens just after the receive
    t_meas, joints = stamp[fresh] + offset, joints[fresh]

    t_cmd = column(commands, "monotonic_ns") / 1e9
    pose_cmd = np.column_stack([column(commands, name) for name in EULER])
    cycle_cmd = column(commands, "cycle").astype(int)
    ev_t = np.array([int(e["monotonic_ns"]) / 1e9 for e in events])
    ev_name = [e["event"] for e in events]
    ev_detail = [e.get("detail", "") for e in events]
    status = [row.get("motion_status", "") for row in piper]
    # --arm-control joint logs the joint target actually sent and its IK diagnostics.
    joint_cmd = np.column_stack([column(commands, f"cmd_joint_{i}") for i in range(1, 7)])
    diagnostics = {name: column(commands, name) for name in ("damping", "step_scale", "pos_err_mm", "rot_err_deg")}
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").is_file() else {}
    return dict(meta=meta, t_meas=t_meas, joints=joints, t_cmd=t_cmd, pose_cmd=pose_cmd, cycle_cmd=cycle_cmd,
                ev_t=ev_t, ev_name=ev_name, ev_detail=ev_detail, poll_s=poll_s, status=status,
                joint_cmd=joint_cmd if np.isfinite(joint_cmd).all() else None, diagnostics=diagnostics)


def derivatives(x, dt):
    return (x, savgol_filter(x, VELOCITY_WINDOW, 3, deriv=1, delta=dt, axis=0),
            savgol_filter(x, ACCELERATION_WINDOW, 3, deriv=2, delta=dt, axis=0))


def held(t, x):
    """Velocity and acceleration of a held-step command sequence, per interval."""
    v = np.diff(x, axis=0) / np.diff(t)[:, None]
    a = np.diff(v, axis=0) / np.diff(t)[1:, None]
    return np.vstack((v, v[-1:])), np.vstack((a, a[-1:], a[-1:]))


def eef_units(pose):
    """XYZ m + Euler rad -> mm + unwrapped degrees."""
    return np.column_stack((pose[:, :3] * 1000, np.degrees(np.unwrap(pose[:, 3:], axis=0))))


def segment_psd(v, masks):
    """Welch PSD averaged over the policy segments (sample-weighted); None if all are too short."""
    psds, weights = [], []
    for mask in masks:
        if mask.sum() >= PSD_SEGMENT:
            f, p = welch(v[mask], fs=GRID_HZ, nperseg=PSD_SEGMENT)
            psds.append(p)
            weights.append(mask.sum())
    return (f, np.average(psds, axis=0, weights=weights)) if psds else None


def band_rms(f, p, low, high):
    keep = (f >= low) & (f < high)
    return float(np.sqrt(np.trapezoid(p[keep], f[keep]))) if keep.sum() > 1 else float("nan")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="output/executor_runs/<timestamp>")
    parser.add_argument("--start", type=float, help="Plot from this many seconds after the first observation")
    parser.add_argument("--end", type=float, help="Plot until this many seconds after the first observation")
    parser.add_argument("--show", action="store_true", help="Also open the figures")
    args = parser.parse_args(argv)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load(args.run)
    fk, bounds = piper_model()
    observe = d["ev_t"][[name == "observe" for name in d["ev_name"]]]
    t0 = observe[0] if len(observe) else d["t_cmd"][0]
    # Default window: policy execution only (the start ramp is in the log too).
    lo = t0 + (args.start or 0.0)
    hi = t0 + args.end if args.end is not None else max(d["t_cmd"][-1], d["ev_t"].max()) + 0.2

    dt = 1 / GRID_HZ
    grid = np.arange(max(lo, d["t_meas"][0]), min(hi, d["t_meas"][-1]), dt)
    q = np.column_stack([np.interp(grid, d["t_meas"], d["joints"][:, j]) for j in range(6)])
    eef = eef_units(np.array([fk(row) for row in q]))
    window = (d["t_cmd"] >= lo) & (d["t_cmd"] <= hi)
    t_cmd, pose_cmd = d["t_cmd"][window], d["pose_cmd"][window]
    # --command-hz above --hz (joint mode): each target is reached by interpolated move_j
    # commands over the following action period, so it is drawn arriving one period later.
    run_args = d["meta"].get("args", {})
    interpolated = (run_args.get("arm_control") == "joint"
                    and round(run_args.get("command_hz", 0) / run_args.get("hz", 1)) > 1)
    shift = 1 / run_args["hz"] if interpolated else 0.0
    if d["joint_cmd"] is not None:
        q_cmd = d["joint_cmd"][window]  # sent by --arm-control joint
    else:  # move_p run: reconstruct with IK seeded at the measured joints
        seeds = np.column_stack([np.interp(t_cmd, d["t_meas"], d["joints"][:, j]) for j in range(6)])
        q_cmd = ik_path(fk, bounds, pose_cmd, seeds)
    eef_cmd = eef_units(pose_cmd)
    # Keep the commanded Euler angles on the same branch as the measured ones.
    for k in range(3, 6):
        eef_cmd[:, k] += 360 * np.round((np.interp(t_cmd, grid, eef[:, k]) - eef_cmd[:, k]) / 360)

    waits, starts, resets = [], [], []
    pending = None
    for t, name in zip(d["ev_t"], d["ev_name"]):
        if name == "observe":
            pending = t
        elif name == "chunk_received" and pending is not None:
            waits.append((pending, t))
            starts.append(t)
            pending = None
        elif name == "pedal_reset":
            resets.append(t)
    # A pedal reset moves the arm back to the start without policy commands, until
    # the next observation; statistics use only the policy segments between resets.
    reset_spans = []
    for t in resets:
        later = observe[observe > t]
        reset_spans.append((t, later[0] if len(later) else hi))
    segments, a = [], lo
    for start, end in reset_spans:
        if start > a:
            segments.append((a, min(start, hi)))
        a = max(a, end)
    if a < hi:
        segments.append((a, hi))
    segments = [(a, b) for a, b in segments if b > a]

    def decorate(ax):
        for a, b in waits:
            if b >= lo and a <= hi:
                ax.axvspan(a - t0, b - t0, color="0.85", lw=0)
        for t in starts:
            if lo <= t <= hi:
                ax.axvline(t - t0, color="0.5", lw=.5, ls=":")
        for a, b in reset_spans:
            if b >= lo and a <= hi:
                ax.axvspan(a - t0, b - t0, color="tab:red", alpha=.12, lw=0)
        ax.grid(alpha=.3)

    measured = {"joints": derivatives(np.degrees(q), dt), "eef": derivatives(eef, dt)}
    commanded = {"joints": (np.degrees(q_cmd), *held(t_cmd, np.degrees(q_cmd))),
                 "eef": (eef_cmd, *held(t_cmd, eef_cmd))}
    labels = {"joints": [f"J{j} (deg)" for j in range(1, 7)], "eef": list(EEF_LABELS)}
    units = {"joints": ("deg", "deg/s", "deg/s²"), "eef": ("", "/s", "/s²")}
    for kind in ("joints", "eef"):
        fig, axes = plt.subplots(6, 3, figsize=(18, 15), sharex=True)
        for row in range(6):
            for col, title in enumerate(("position", "velocity", "acceleration")):
                ax = axes[row, col]
                decorate(ax)
                ax.plot(grid - t0, measured[kind][col][:, row], lw=.8, color="tab:blue", label="measured")
                for k, (a, b) in enumerate(segments):  # no commanded line through pedal resets
                    m = (t_cmd >= a) & (t_cmd <= b)
                    style = dict(lw=.8, color="tab:orange", label="commanded" if k == 0 else None)
                    if interpolated and col == 0:  # ramps that arrive one action period after issue
                        ax.plot(t_cmd[m] + shift - t0, commanded[kind][col][m, row], **style)
                    else:
                        ax.step(t_cmd[m] + shift - t0, commanded[kind][col][m, row], where="post", **style)
                if row == 0:
                    ax.set_title(title)
                if col == 0:
                    ax.set_ylabel(labels[kind][row])
                elif kind == "joints":
                    ax.set_ylabel(units[kind][col])
        for ax in axes[-1]:
            ax.set_xlabel("s from first observation")
        axes[0, 0].legend(loc="upper right", fontsize=8)
        fig.suptitle(f"{args.run.name} {kind}: measured (200 Hz; Savitzky-Golay {VELOCITY_WINDOW / GRID_HZ * 1000:.0f} ms "
                     f"velocity, {ACCELERATION_WINDOW / GRID_HZ * 1000:.0f} ms acceleration) vs commanded "
                     f"({'interpolated, arriving one action later' if interpolated else 'held steps'}); "
                     "gray = waiting for predictor, red = pedal reset")
        fig.tight_layout()
        fig.savefig(args.run / f"{kind}.png", dpi=110)
        print(f"Saved {args.run / f'{kind}.png'}")

    # Sample tracking error at target arrival, matching the timing in eef.png.
    # Keep samples within feedback coverage and exclude moves across pedal resets.
    arrival = t_cmd + shift
    error = np.full((len(t_cmd), 3), np.nan)
    for a, b in segments:
        valid = ((t_cmd >= a) & (arrival <= b)
                 & (arrival >= grid[0]) & (arrival <= grid[-1]))
        measured_xyz = np.column_stack([
            np.interp(arrival[valid], grid, eef[:, j]) for j in range(3)
        ])
        error[valid] = measured_xyz - eef_cmd[valid, :3]
    distance = np.linalg.norm(error, axis=1)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for j, ax in enumerate(axes[:3]):
        decorate(ax)
        ax.plot(arrival - t0, error[:, j], lw=.9)
        ax.axhline(0, color="0.4", lw=.6)
        ax.set_ylabel(f"{'XYZ'[j]} error (mm)")
    decorate(axes[3])
    axes[3].plot(arrival - t0, distance, color="tab:purple", lw=.9)
    axes[3].set_ylabel("3D error (mm)")
    axes[3].set_xlabel("s from first observation")
    finite = distance[np.isfinite(distance)]
    if len(finite):
        axes[3].set_title(f"Mean {finite.mean():.2f} mm; RMS {np.sqrt(np.mean(finite ** 2)):.2f} mm; "
                          f"95th percentile {np.percentile(finite, 95):.2f} mm; max {finite.max():.2f} mm",
                          fontsize=10)
    fig.suptitle(f"{args.run.name}: EEF position error (measured FK minus requested target)\n"
                 f"Sampled {'one action period after issue' if interpolated else 'at command issue'}; "
                 "gray = waiting for predictor, red = pedal reset")
    fig.tight_layout()
    fig.savefig(args.run / "eef_position_error.png", dpi=110)
    print(f"Saved {args.run / 'eef_position_error.png'}")

    # Spectrum: measured velocity vs the commanded path interpolated linearly (its intended motion).
    cmd_fs = 1 / np.median(np.diff(t_cmd))
    fig, axes = plt.subplots(2, 6, figsize=(22, 6.5), sharex=True)
    masks = [(grid >= a) & (grid <= b) for a, b in segments]
    policy_s = sum(m.sum() for m in masks if m.sum() >= PSD_SEGMENT) / GRID_HZ
    if not policy_s:
        raise SystemExit("No policy segment of at least 2 s between pedal resets in this window")
    print(f"\nVelocity RMS by band, measured / commanded (command rate {cmd_fs:.1f} Hz; "
          f"{policy_s:.0f} s of policy in {len(segments)} segments, pedal resets excluded):")
    print("            " + "".join(f"{f'{a:g}-{b:g} Hz':>18}" for a, b in BANDS_HZ) + "   peak 3-20 Hz")
    for r, kind in enumerate(("joints", "eef")):
        # Intended path: commands interpolated linearly, within each segment only.
        smooth_cmd = np.full((len(grid), 6), np.nan)
        for (a, b), mask in zip(segments, masks):
            m = (t_cmd >= a) & (t_cmd <= b)
            if m.sum() >= 2:
                for j in range(6):
                    smooth_cmd[mask, j] = np.interp(grid[mask], t_cmd[m] + shift, commanded[kind][0][m, j])
        v_cmd = np.gradient(np.nan_to_num(smooth_cmd), dt, axis=0)
        for j in range(6):
            v = np.gradient(measured[kind][0][:, j], dt)
            f, p = segment_psd(v, masks)
            fc, pc = segment_psd(v_cmd[:, j], masks)
            ax = axes[r, j]
            ax.semilogy(f, p, lw=.8, label="measured")
            ax.semilogy(fc, pc, lw=.8, label="commanded")
            ax.axvline(cmd_fs, color="0.5", ls=":", lw=.8)
            ax.set_xlim(0, 40)
            shown = np.r_[p[(f > 0) & (f < 40)], pc[(fc > 0) & (fc < 40)]]
            ax.set_ylim(np.percentile(shown, 2) / 3, shown.max() * 3)
            ax.set_title(labels[kind][j])
            ax.grid(alpha=.3)
            cells = "".join(f"{band_rms(f, p, a, b):8.2f}/{band_rms(fc, pc, a, b):<8.2f}  " for a, b in BANDS_HZ)
            above = (f > 3) & (f < 20)  # higher peaks (~57 Hz) are feedback timing artifacts
            print(f"{labels[kind][j]:<12}{cells}{f[above][np.argmax(p[above])]:6.1f} Hz")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Hz (dotted: command rate)")
    fig.suptitle(f"{args.run.name}: velocity PSD, measured vs commanded")
    fig.tight_layout()
    fig.savefig(args.run / "spectrum.png", dpi=110)
    print(f"\nSaved {args.run / 'spectrum.png'}")

    # Does MOVE P finish each target before the next one arrives (stop-and-go)?
    in_chunk = np.zeros(len(d["poll_s"]), bool)
    for a, b in zip(starts, [t for t, n in zip(d["ev_t"], d["ev_name"]) if n == "chunk_done"]):
        in_chunk |= (d["poll_s"] >= a) & (d["poll_s"] <= b)
    reached = np.array(["SUCCESS" in s for s in d["status"]])
    if in_chunk.any():
        toggles = np.sum(np.diff(reached[in_chunk].astype(int)) == 1)
        seconds = in_chunk.sum() / GRID_HZ
        print(f"motion_status 'reached target' while executing chunks: {reached[in_chunk].mean() * 100:.0f}% of "
              f"samples, {toggles / seconds:.1f} times/s (command rate {cmd_fs:.1f} Hz)")
    # Chunk boundaries: first commanded step of each chunk vs steps inside chunks.
    steps = np.linalg.norm(np.diff(pose_cmd[:, :3], axis=0), axis=1) * 1000
    boundary = np.diff(d["cycle_cmd"][window]) != 0
    across_reset = np.array([any(t_cmd[i] < r < t_cmd[i + 1] for r in resets) for i in range(len(steps))], bool)
    steps, boundary = steps[~across_reset], boundary[~across_reset]
    if boundary.any() and (~boundary).any():
        print(f"commanded flange step: inside chunks median {np.median(steps[~boundary]):.1f} mm, "
              f"at chunk boundaries median {np.median(steps[boundary]):.1f} / max {steps[boundary].max():.1f} mm")
    # Observation state anchor (--state-anchor): "command; measured off X mm Y deg" or a measured fallback.
    notes = [detail for t, name, detail in zip(d["ev_t"], d["ev_name"], d["ev_detail"])
             if name == "state" and lo <= t <= hi]
    if notes:
        anchored = [n for n in notes if n.startswith("command")]
        offsets = [float(n.split("off ")[1].split(" mm")[0]) for n in notes if "off " in n]
        print(f"observation state: commanded pose on {len(anchored)}/{len(notes)} observations, "
              f"{sum('FALLBACK' in n for n in notes)} measured fallbacks; arm-to-command offset median "
              f"{np.median(offsets) if offsets else float('nan'):.1f} mm, max {max(offsets, default=float('nan')):.1f} mm")
    if d["joint_cmd"] is not None:
        diag = {name: values[window] for name, values in d["diagnostics"].items()}
        print(f"joint control: IK damped on {np.mean(diag['damping'] > 0) * 100:.0f}% of commands, speed-capped on "
              f"{np.mean(diag['step_scale'] < 1) * 100:.0f}%; sent target vs policy pose median "
              f"{np.median(diag['pos_err_mm']):.2f} mm / {np.median(diag['rot_err_deg']):.2f} deg, "
              f"max {diag['pos_err_mm'].max():.1f} mm / {diag['rot_err_deg'].max():.1f} deg")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
