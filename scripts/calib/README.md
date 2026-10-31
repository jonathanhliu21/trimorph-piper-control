Record while the Quest controller is driving the PiperX arm:

```bash
python scripts/calib/coordinate_record.py --output output/coords.csv --duration 30 --piper-channel can0 --quest-port 5005
python scripts/calib/get_coord_transform.py output/coords.csv --output output/transform.json --model similarity
```

For foot pedal start/stop, omit `--duration`:

```bash
python scripts/calib/coordinate_record.py --output output/coords.csv --piper-channel can0 --quest-port 5005
```

After both workers initialize, press the pedal to start calibration and press it again to stop and save. The pedal is auto-detected; use `--foot-pedal-device /dev/input/...` to select it explicitly. Optional `--foot-pedal-key-code` and `--foot-pedal-debounce-seconds` (default 0.5) configure key filtering and debounce. Ctrl-C before the first press exits without saving a CSV; during recording it saves available samples.

Combine multiple recordings into one shared calibration:

```bash
python scripts/calib/get_coord_transform.py output/calib/coords.csv output/calib/coords2.csv \
  --output output/calib/transform.json --model similarity -f
```

Each CSV contributes equal total weight by default (`--weighting equal`). Use `--weighting samples` to give every sample equal weight instead. The script jointly fits relative-motion data, preserving valid rotations rather than taking an elementwise matrix average. It centers positions separately and computes rotation changes from each file's own starting orientation; it never treats file boundaries as motion. Different position origins and fixed tool orientation offsets are allowed, but world axes, units, and the teleoperation mapping must remain consistent between recordings. The combined motion must sufficiently excite all position axes and multiple rotation axes.

The JSON includes `input_csvs`, combined errors, and per-file errors/references in `recordings`. For multiple files, the top-level absolute position matrix, reference positions, and offset are `null` because each recording has its own origin; the relative matrices remain usable by the chunk helpers. Single-file invocation retains its absolute-position diagnostics.

The recorder uses the collection Piper and Quest workers and sends no motion commands. Start the Quest pose sender first. If teleoperation already receives on port 5005, duplicate/forward its packets to a separate port and select that with `--quest-port`; two receivers cannot independently consume the same UDP stream. Packets use the existing `seq, px, py, pz, qx, qy, qz, qw` JSON schema.

With an explicit `--duration`, recording starts automatically after both workers initialize. `--duration 0` records until Ctrl-C; Ctrl-C also saves an early stop. Translate slowly through a substantial 3D trajectory, then rotate gently about at least two nonparallel axes (ideally all three), with a few combined rotations. Stay within the arm's comfortable workspace. Orientation must actually track the controller during this recording. A position-only teleoperation mode cannot provide angular calibration. The position fitter rejects nearly planar, straight-line, or stationary trajectories; the orientation fitter rejects stationary or single-axis rotations. Existing output files are preserved by default; pass `-f` / `--force` to either script to overwrite its output.

The CSV columns are `sample_idx, monotonic_ns, elapsed_s, piper_x, piper_y, piper_z, quest_x, quest_y, quest_z, piper_qx, piper_qy, piper_qz, piper_qw, quest_qx, quest_qy, quest_qz, quest_qw`. Quaternions use xyzw order. Piper roll/pitch/yaw are interpreted in radians as `Rz(yaw) @ Ry(pitch) @ Rx(roll)`, matching the dataset postprocessor. Quest quaternions are normalized, with no axis changes before fitting. Position values preserve native units and axes. Older XYZ-only CSVs remain usable with `get_coord_transform.py --position-only`; they cannot recover orientation.

The recorder uses host monotonic timestamps, linear interpolation for positions, and shortest-path quaternion SLERP for orientations (including Euler wrapping and quaternion sign changes). It retains only stream overlap and excludes interpolation brackets over `--max-gap-s` (default 0.1 seconds). Default output rate is 50 Hz. EpisodeBuffer itself only buffers raw data. Interpolation does not remove network delays, controller latency, or arm tracking error.

For column vectors, position calibration fits `A` after centering the trajectories:

```python
delta_piper = A @ delta_quest
piper_target = piper_start + A @ (quest_current - quest_start)
```

Choose the position model to match the teleoperation mapping:

- Default `--model rigid`: proper rotation with unit gain.
- `--model similarity`: rotation plus fitted uniform gain, for motion scaling or different units.
- `--model linear`: general 3x3 mapping including separate axis gains, reflection, and shear. Requires well-excited motion and can overfit tracking errors.
- `--allow-reflection`: permit opposite handedness for rigid/similarity position fits, for example if the sender's coordinates have not already been converted to the robot's handedness.

If the tracked controller point differs from the robot flange point, add `--fit-offset`:

```bash
python scripts/calib/get_coord_transform.py output/calib/coords_t.csv output/calib/coords_r.csv \
  --output output/calib/transform_with_offset.json --model similarity --allow-reflection --fit-offset
```

This jointly fits `A` and a constant correction vector `l` in Piper flange coordinates:

```python
piper = A @ quest + R_piper @ l + origin_offset
piper_target = piper_start + A @ (quest_current - quest_start) + (Rp_target - Rp_start) @ l
```

The sign convention is **add** `R_piper @ l` to the mapped Quest point to obtain the Piper flange point. The vector is saved as `tracked_point_offset.vector` in schema version 3, in Piper position units (meters with pyAgxArm). It is a rotating correction, separate from the constant origin offset. The fit needs independent 3D translation and rotations about multiple axes; recordings can be separate files. Degenerate offset fits are rejected. `--position-only` cannot be combined with `--fit-offset`.

`transform_chunk` and `map_relative_pose` apply this correction automatically. With an offset calibration, `map_relative_pose` requires `piper_start_quaternion` even for world-frame deltas. Dataset postprocessing requires `--piper-start-pose` with the actual starting XYZ/RPY: the correction depends on flange orientation, so a trajectory corrected assuming identity orientation cannot simply be reanchored at a different starting orientation later.

The CLI reports position RMSE using measured flange orientations, position RMSE using mapped orientations after centering each recording, and position RMSE anchored at each recording's first measured pose. The last value includes the effect of the orientation mapping and start-sample error. Fit errors are training diagnostics; validate on held-out motion. With `--fit-offset`, `position_matrix_4x4` is `null` even for one CSV because one fixed affine position transform cannot express the rotating correction. `displacement_matrix_4x4` contains only the linear component; use the pose helpers for the full conversion.

Orientation has its own independently fitted proper rotation `B`:

```python
Dq = Rq_current @ Rq_start.T
Dp = B @ Dq @ B.T
Rp_target = Dp @ Rp_start
```

These are spatial relative rotations. A constant controller/tool orientation offset cancels when forming `Dq` and `Dp`. The fit aligns rotation vectors, preserves rotation angles, and reports angular residuals; it does not fit angular gain. Never subtract Euler angles or apply position scale/shear to orientation. Even if the positional basis `C` is reflected, the axial basis for rotation vectors is the proper rotation `det(C) * C`. Position and angular mappings may differ in a teleoperation implementation, so `B` is fitted from the orientation data rather than assumed from `A`.

The JSON includes `linear_matrix` (`A`), a zero-translation `displacement_matrix_4x4`, and `orientation.rotation_matrix` (`B`). `position_matrix_4x4` includes an offset for the recorded absolute origins and is for absolute-position diagnostics; chunk conversion does not use it. Position and orientation mappings together are not necessarily a single rigid SE(3) transform.

For an absolute Quest pose chunk, use the helper to generate absolute Piper targets anchored at the actual Piper chunk-start pose:

```python
import json
from scripts.calib.get_coord_transform import transform_chunk

with open("output/transform.json") as stream:
    calibration = json.load(stream)

piper_xyz, piper_xyzw = transform_chunk(
    quest_positions,          # Nx3 absolute positions
    quest_quaternions,        # Nx4 absolute orientations, xyzw
    piper_start_position,     # 3 values at this chunk's beginning
    piper_start_quaternion,   # 4 values, xyzw, at this chunk's beginning
    calibration,
)
```

The first output pose equals the supplied Piper starting pose. Each target uses displacement from the first Quest pose, so there is no per-step integration. Reset both anchors at every chunk boundary.

If your policy already produces deltas, choose the convention explicitly:

```python
from scripts.calib.get_coord_transform import map_relative_pose

# World axes: dp = p(t)-p(start), D = R(t) @ R(start).T.
piper_dp, piper_dq = map_relative_pose(
    quest_dp, quest_dq, calibration, piper_start_quaternion=piper_start_quaternion,
)
# Absolute targets: piper_start + piper_dp, Rotation(piper_dq) * Rotation(piper_start_quat).

# Chunk-start axes: dp = R(start).T @ (p(t)-p(start)), D = R(start).T @ R(t).
piper_local_dp, piper_local_dq = map_relative_pose(
    quest_local_dp, quest_local_dq, calibration,
    relative_frame="chunk",
    quest_start_quaternion=quest_start_quaternion,
    piper_start_quaternion=piper_start_quaternion,
)
# Absolute targets: piper_start + Rp_start @ piper_local_dp,
#                   Rp_start @ Rotation(piper_local_dq).as_matrix().
```

Inputs and outputs can be one pose or matching batches; quaternion deltas also use xyzw. Chunk-local mapping needs both starting orientations because positions and rotations must be expressed in the appropriate axes before and after calibration. These helpers produce targets only; they do not command the arm.

Fit diagnostics include position RMSE in Piper position units, consecutive-displacement RMSE, angular RMSE/max in radians, motion singular values, gains, and determinant. These are errors on the supplied recording; validate on a separate trajectory. A poor fit may reflect latency, tracking limits, insufficient motion, angular gain, or an unsuitable position model. By default the calibration does not estimate a controller-to-flange lever arm; use `--fit-offset` when changing orientation introduces position error because the tracked points differ.

Inspect recorded controller motion with CSV playback:

```bash
python scripts/quest/visualize_quest.py --from-csv output/calib/coords.csv --speed 0.5
```

The viewer shows the 3D trajectory, current orientation axes, and XYZ time traces. Use Space or the Play/Pause button, Left/Right to step individual samples while paused, the time slider or a click on the traces to seek, and Previous/Next jump to inspect flagged position steps. Steps over `--jump-threshold` (default 0.05 incoming position units) are highlighted in red; this is a review threshold, not a tracking-failure diagnosis. Playback accepts both calibration CSVs and raw Quest worker CSVs, preserves recorded sample values, and opens no UDP receiver. Run without `--from-csv` for the existing live view.

In live mode, the same `--jump-threshold` counts significant position steps between consecutive received packets, including steps occurring between screen refreshes. A red alert and controller marker flash for 1.5 seconds; a dashed red segment marks the latest jump. The display retains the session count and the last jump's distance, packet interval, and elapsed time. Each above-threshold step counts once (an out-and-back counts twice); restarting the viewer resets the count. The threshold is in incoming position units and is not a velocity or singularity check.

```bash
python scripts/quest/visualize_quest.py --jump-threshold 0.05
```

The coordinate recorder also reports position jumps in the terminal during recording:

```bash
python scripts/calib/coordinate_record.py --output output/calib/coords.csv --jump-threshold 0.05
```

Every consecutive raw Quest packet and Piper sample is checked before interpolation. Steps strictly above the threshold print a flushed `WARNING: [quest] POSITION JUMP` or `[piper]` message with a per-device count, time since recording began, distance, sample interval, and XYZ displacement. A final summary prints both counts. The threshold is in each device's native position units; all samples remain in the recording, and no filtering or calibration changes are applied.

### Visualize ROS 2 mocap detections

Source your ROS 2 installation and Vicon receiver workspace, then run:

```bash
python scripts/calib/mocap_detection.py
# Select a specific topic and an arrow length in the receiver's position units:
python scripts/calib/mocap_detection.py --mocap-topic /your/pose/topic --axis-length 50
```

The interactive 3D plot automatically discovers `vicon_receiver/msg/Position`
and `vicon_receiver/msg/MarkerPositions` topics. Repeat `--mocap-topic` (or
`--topic`) to select several topics. Segment poses appear as labeled black dots
with red/green/blue local X/Y/Z orientation arrows. Marker positions appear as
orange dots without orientation. Zero or invalid quaternions show position only
and an “orientation unavailable” label.

Coordinates and translation units are preserved from the receiver, with no
coordinate frame transforms. The status line shows source frame IDs and
`translation_type`; select topics in the same coordinate frame and units when
comparing objects. Occluded markers and nonfinite positions are hidden, and
objects disappear after two seconds without an update (`--stale-after-s`).
`--refresh-hz` controls display refresh (default 20); `--axis-length` defaults
to an automatic display scale. Drag to rotate the view. Close the window or
press Ctrl-C to stop and release the ROS subscriptions. This viewer does not
command hardware or write a CSV.

### Rectangle repeatability: Piper FK vs Vicon / Quest

Commands the arm (it moves). Start with the tracked marker or controller rigidly
mounted on the flange and the arm at a safe pose with ~20 mm of clearance in the
chosen plane; that pose becomes the rectangle's first corner.

```bash
# Vicon marker on the arm (uses .venv-mocap + ros2/mocap_env.sh automatically):
python scripts/calib/rectangle_repeatability.py --label mocap
# Quest controller on the arm, Quest sender streaming to UDP 5005 (add --no-mocap if Vicon is off):
python scripts/calib/rectangle_repeatability.py --label quest --quest-port 5005
python scripts/calib/analyze_rectangle.py output/rect_repeatability/<timestamp>_<label>
```

Defaults: 20 loops of a 20 x 20 mm rectangle in the Piper base `xy` plane at
10 mm/s with a 1 s dwell per corner (about 4 minutes), orientation held
constant, targets streamed with `move_eef` at 50 Hz. The script prints the
corners and waits for Enter. Streaming stops if the reported flange pose
deviates from the target by more than `--max-tracking-error-mm` (default 10).
Each run directory holds `piper.csv` (flange pose + joints), `commands.csv`,
`mocap.csv`, `quest.csv` and `meta.json`, all with host monotonic timestamps.

The analyzer finds each tracker's time lag, rigidly aligns it to Piper FK,
and reports residual RMSE/max (moving vs settled), per-loop RMSE, corner
repeatability across loops and edge lengths, plus FK vs commanded error. It
writes `summary.json` and `analysis.png` in the run directory. Rigid alignment
absorbs the tracker frame and the marker's constant offset from the flange,
not scale or shape errors.

### Quest calibration sequence (shapes, rotations, dynamic trajectory)

Commands the arm. Mount the Quest controller rigidly on the flange, start the
Quest sender, then:

```bash
python scripts/calib/quest_calibration_sequence.py --list                    # show the plan (no hardware)
python scripts/calib/quest_calibration_sequence.py --quest-port 5005 --no-mocap
python scripts/calib/fit_quest_calibration.py output/quest_calibration/<timestamp>_quest_calib
python scripts/calib/quest_calibration_sequence.py --home-only               # move to home only
```

The arm moves to a fixed home joint configuration (`--home-joints-deg`) in joint
space before the sequence, before each segment and at the end. Default segments
(about 4 min of motion): 30 s warmup hold; a 30 mm rectangle and 30 mm circle in
each of the Piper `xy`, `xz`, `yz` planes; ±10° sweeps of wrist joints 4, 5 and 6;
and 60 s of multi-sine motion on all six joints (dynamic trajectory). Joint-space
segments are checked against joint limits and the wrist singularity before the arm
moves; a Cartesian segment that exceeds `--max-tracking-error-mm` is stopped,
marked failed in `meta.json`, and the sequence continues from home. Select
segments with `--segments`, and drop Vicon with `--no-mocap` (keep it to also record
Vicon ground truth).

`fit_quest_calibration.py` estimates the Quest latency, fits rigid / similarity /
linear Quest-to-Piper models with and without the rotating controller-to-flange
offset (via `get_coord_transform.py`, each segment as its own recording), scores
them on held-out segments (`--holdout`, default `rect_xy circle_xz`) with
`transform_chunk` anchored at each segment start, and saves the best model refitted
on all segments to `<run>/calibration/quest_calibration.json` (plus `latency_s`),
with `model_comparison.json`, leave-one-segment-out errors and `report.png`.
Apply it with `transform_chunk` after shifting Quest timestamps by `latency_s`. A
calibration only holds for the Quest tracking session it was recorded in.

### Quest teleoperation with the calibration

`scripts/teleop/quest_teleop.py` drives Piper from Quest deltas using a
`quest_calibration.json`. It first moves to the fixed home joints (same default as
the calibration sequence), then, while the foot pedal is held, maps Quest motion
since the moment of engagement onto the robot target (position only by default,
`--with-orientation` to follow rotation). Releasing holds the arm; re-engaging
re-anchors at the current target.

```bash
CAL=output/quest_calibration/<run>/calibration_quest/quest_calibration.json
python scripts/teleop/quest_teleop.py --calibration $CAL --quest-port 5005 --dry-run   # no arm, print targets
python scripts/teleop/quest_teleop.py --calibration $CAL --quest-port 5005
```

Targets are limited to `--max-offset-mm` (100) from home, `--max-rotation-deg`
(45), `--max-speed-mm-s` (300) and `--max-angular-speed-deg-s` (180). The clutch
disengages on stale Quest data, a Quest jump, or a tracking error above
`--max-tracking-error-mm`. Type `h` + Enter to go home, `q` + Enter to go home and
quit; Ctrl-C stops streaming without moving. `--no-pedal` uses Enter as a toggle.
Each session is logged to `output/teleop/<timestamp>/`.

### Record a Quest demonstration and replay it in absolute coordinates

`scripts/teleop/quest_replay.py` maps every Quest sample to an absolute flange
target with the full calibration, `p = A q + R l + c` (`c` is the origin of the
calibration recordings), so the arm goes where the hand-held controller was rather
than moving by deltas from a clutch anchor. Orientation is held at home by default;
`--with-orientation` adds the controller's rotation since the first sample.

```bash
CAL=output/quest_calibration/<run>/calibration_quest/quest_calibration.json
python scripts/teleop/quest_replay.py --calibration $CAL --quest-port 5005 --dry-run   # no arm: record + check
python scripts/teleop/quest_replay.py --calibration $CAL --quest-port 5005
python scripts/teleop/quest_replay.py --calibration $CAL --replay output/quest_replay/<session>/rec_01
```

The arm moves to home and holds still while you record (pedal press = start, press
again = stop; `--no-pedal` uses Enter). The status line shows where the arm would
go relative to home. A recording is replayed only if it stays within
`--max-offset-mm` (150) and `--max-rotation-deg` (30) of home, under
`--max-speed-mm-s` / `--max-angular-speed-deg-s` at `--playback-speed`, and has no
Quest gap or jump. Before each replay the arm moves to the first target along a
minimum-jerk path (`--approach-speed-mm-s`, 50), holds it for `--settle-s` and
checks it within `--start-tolerance-mm`. The playback rate eases in and out over
`--ease-s`, so the arm starts and stops at rest. Streaming stops on the same
faults as teleop. The origin `c` is only valid while the Quest keeps the
calibration's tracking session; it drifted about 1 cm over 30 min on 2026-10-31.
Recordings are saved to `output/quest_replay/<session>/rec_NN/`.
