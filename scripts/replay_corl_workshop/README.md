# Transformed Piper coordinate viewer

Run from the repository root using an environment with `requirements.txt` installed:

```bash
.venv/bin/python scripts/replay_corl_workshop/replay_transformed_piper.py --episode 3
```

The default source is `data_trimorph_pingpong_pick_place_mocap/absolute_action_t`.
The viewer shows the Piper-base XYZ path, the selected end-effector coordinate
axes (red X, green Y, blue Z), XYZ traces, and six estimated joint-angle traces.
Use the frame slider, **Play/Pause**, and **Next issue** to inspect the trajectory.
Close the window to continue to the next episode when processing a directory.

Scan all episodes without opening windows, or select another dataset/file:

```bash
.venv/bin/python scripts/replay_corl_workshop/replay_transformed_piper.py --no-show
.venv/bin/python scripts/replay_corl_workshop/replay_transformed_piper.py \
  data_trimorph_pingpong_pick_place_mocap/absolute_action_t/000003.npy \
  --jump-deg 5 --joint-margin-deg 0
```

Every episode produces `<episode>.csv` (all diagnostic events), `.npz` (poses,
estimated joints in radians, timestamps, IK errors, bounds and seed), and `.png`
in `output/replay_transformed_piper/`. Change this with `--output-dir`.
The console prints the first 30 events; `--print-limit -1` prints all events.
Frame numbers are **zero-based rows of the action .npy**, and times are seconds
from the first pose timestamp in the matching `syncs/*.npz`. They are not camera
frame numbers. Every source row is analyzed, without resampling or hand trimming.
Standalone `.npy` inputs require `--hz RATE --pose-frame piper_base`.

Use `--action-file` (alias `--actions`) to explicitly select a file from another dataset:

```bash
.venv/bin/python scripts/replay_corl_workshop/replay_transformed_piper.py \
  --action-file data_trimorph_pick_place_calib/absolute_action_t/000000.npy
```

For files in an `absolute_action_t` folder, timing and pose-frame metadata come
from that dataset's sibling `syncs` folder. For a standalone file, add
`--hz RATE --pose-frame piper_base`. Use either a positional source or
`--action-file`; `--episode` only applies to a source directory.

The script reuses `data_collection/replay_actions.py`'s loader, Piper SDK model,
limits, IK residual convention, and 0.5 mm / 0.2 degree fit tolerances. Defaults
flag joint jumps above 2 degrees between consecutive source rows, a 3 degree
joint-bound margin, a 10 degree wrist margin, end-effector rotation jumps above
10 degrees, and position jumps above 0.15 m. `joint_limit` means the estimated
angle exceeds an SDK bound; `joint_margin` means it only exceeds the inset
allowed range. Jump rows give the joint, previous/current frame and angles,
change in degrees, and threshold. An inaccurate local fit produces `ik_failure`
with residuals, leaving that frame's joints unavailable; scanning continues.

Joint estimates depend on the IK seed. Set `--start-joints-deg J1 J2 J3 J4 J5 J6`
to a known starting configuration; the offline default is `0 45 -45 0 30 0`.
The first pose uses bounded initialization, then an unconstrained continuous
branch is followed to expose angles beyond the limits. Equivalent full turns
are placed nearest the previous solution. A violation on that branch does not
prove that every alternative IK branch violates a limit. No robot is connected,
and the estimates are not measured angles or a validated executable plan.
