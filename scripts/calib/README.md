Record while the Quest controller is driving the PiperX arm:

```bash
python scripts/calib/coordinate_record.py --output output/coords.csv --duration 30 --piper-channel can0 --quest-port 5005
python scripts/calib/get_coord_transform.py output/coords.csv --output output/transform.json --model similarity
```

Combine multiple recordings into one shared calibration:

```bash
python scripts/calib/get_coord_transform.py output/calib/coords.csv output/calib/coords2.csv \
  --output output/calib/transform.json --model similarity -f
```

Each CSV contributes equal total weight by default (`--weighting equal`). Use `--weighting samples` to give every sample equal weight instead. The script jointly fits relative-motion data, preserving valid rotations rather than taking an elementwise matrix average. It centers positions separately and computes rotation changes from each file's own starting orientation; it never treats file boundaries as motion. Different position origins and fixed tool orientation offsets are allowed, but world axes, units, and the teleoperation mapping must remain consistent between recordings. The combined motion must sufficiently excite all position axes and multiple rotation axes.

The JSON includes `input_csvs`, combined errors, and per-file errors/references in `recordings`. For multiple files, the top-level absolute position matrix, reference positions, and offset are `null` because each recording has its own origin; the relative matrices remain usable by the chunk helpers. Single-file invocation retains its absolute-position diagnostics.

The recorder uses the collection Piper and Quest workers and sends no motion commands. Start the Quest pose sender first. If teleoperation already receives on port 5005, duplicate/forward its packets to a separate port and select that with `--quest-port`; two receivers cannot independently consume the same UDP stream. Packets use the existing `seq, px, py, pz, qx, qy, qz, qw` JSON schema.

Recording starts after both workers initialize. `--duration 0` records until Ctrl-C; Ctrl-C also saves an early stop. Translate slowly through a substantial 3D trajectory, then rotate gently about at least two nonparallel axes (ideally all three), with a few combined rotations. Stay within the arm's comfortable workspace. Orientation must actually track the controller during this recording. A position-only teleoperation mode cannot provide angular calibration. The position fitter rejects nearly planar, straight-line, or stationary trajectories; the orientation fitter rejects stationary or single-axis rotations. Existing output files are preserved by default; pass `-f` / `--force` to either script to overwrite its output.

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
piper_dp, piper_dq = map_relative_pose(quest_dp, quest_dq, calibration)
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

Fit diagnostics include position RMSE in Piper position units, consecutive-displacement RMSE, angular RMSE/max in radians, motion singular values, gains, and determinant. These are errors on the supplied recording; validate on a separate trajectory. A poor fit may reflect latency, tracking limits, insufficient motion, angular gain, or an unsuitable position model. This calibration does not estimate a controller-to-flange lever arm: changing orientation can introduce position error if the points tracked by the two devices differ and teleoperation does not compensate for that offset.

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
