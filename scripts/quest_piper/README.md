# Quest + Piper without a hand

This workflow records a Quest controller rigidly attached to the Piper flange,
compares calibrated Quest poses with recorded joint-feedback FK, and replays
the Quest-derived trajectory while measuring agreement with the original arm
motion. No hand, camera, or Vicon is required.

Run commands from the repository root in the collection environment (NumPy,
SciPy, Matplotlib, OpenCV and pyAgxArm). Keep the controller mounting rigid and
the Quest tracking session unchanged from calibration. Stop other receivers on
UDP 5005 before collection. Replay reads recorded targets, so it needs no UDP
stream.

## 1. Collect both streams passively

```bash
python scripts/quest_piper/collect.py \
  --dataset-root data_quest_piper \
  --calibration-json output/quest_calibration/20261108_142941_quest_calib/calibration_quest/quest_calibration.json \
  --quest-port 5005 --channel can0
```

Move the attached arm/controller using the arm's built-in drag-teach mode.
Collection never enables the arm, changes payload/installation settings, or
sends motion commands. Foot-pedal control is the default, matching
`collect_hand_only.py`: press once to start after a **0.2-second ready delay**;
press again during that delay to cancel. During recording, **press and release
to save**, or **hold for 2 seconds to discard** the episode. **Q/Esc** in the viewer or `q` then Enter stops collection
and saves the active episode. Re-run with the same root to append episodes.

Use `--episode-s 30` for one timed episode (automatically disables the pedal),
or `--no-pedal` for Space/Enter start/stop. `--pedal` / `--use-foot-pedal` explicitly
select the default pedal mode. `--foot-pedal-device` selects the input device;
`--foot-pedal-key-code`, `--foot-pedal-debounce-seconds` (default 0.5),
`--ready-seconds`, and `--hold-delete-s` configure its behavior. `--no-view` enables headless
recording. Ctrl-C saves the active episode.

The same Quest/Piper accuracy viewer opens by default: orange Quest-derived
flange, cyan measured FK, local XYZ axes, position/angular errors, IK warnings,
and orange X position-jump markers. **A** re-anchors the viewer; **R** clears its
statistics. If no calibration is supplied, **C** captures a viewer-only fit.
Viewer settings/fits never alter raw data or choose the calibration used by
postprocessing. The REC/READY banner shows recording state. C/A/R affect only
the viewer. Space/Enter control episodes only with `--no-pedal`.

Raw layout follows `collect_hand_only.py`:

```text
DATASET/raw/episode_000000/
  quest.csv          # sample_idx, monotonic_ns, wall_time_s, elapsed_s, seq, px/py/pz, qx/qy/qz/qw
  piper.csv          # sample_idx, monotonic_ns, wall_time_s, elapsed_s, x/y/z, roll/pitch/yaw, joint_1..6, ...
  metadata.json
```

Positions are meters, Euler/joint angles radians, quaternions XYZW. Here the
Piper `x/y/z/roll/pitch/yaw` columns explicitly mean **joint-feedback flange
FK**, consistent with calibration; duplicate `fk_*` columns and optional
`fw_*` firmware poses are retained for auditing. A repeated SDK joint timestamp
is not recorded as a new sample. Metadata identifies FK as the reference and
marks the hand/camera absent. No fake hand CSV or encoder arrays are created.
The existing hand-dependent postprocessor is not used for these episodes.

## 2. Postprocess using the desired calibration

```bash
python scripts/quest_piper/postprocess.py data_quest_piper \
  --calibration-json output/quest_calibration/20261108_142941_quest_calib/calibration_quest/quest_calibration.json
```

Use `--episode 0` to select an episode (repeat for several), and `--overwrite`
to regenerate outputs with another calibration. Raw files are preserved.

The saved `latency_s` is applied by default: Quest received at `t` is compared
with Piper FK at `t - latency_s`. Use `--latency-s 0` to disable that shift, or
provide another known value. Positions are linearly interpolated, rotations use
SLERP, and no extrapolation is permitted. End samples without overlap are
trimmed. Interior gaps over `--max-gap` (default 50 ms) are rejected rather than
silently bridged into a replay path.

Default conversion reuses `get_coord_transform.transform_chunk`, including the
rotating flange offset. It anchors once to FK at the first synchronized pose.
Therefore initial position/orientation errors are zero by construction and
constant origin bias is removed; subsequent errors measure relative tracking
agreement. There is no per-frame or per-segment refitting.

For absolute calibration evaluation, add **`--absolute`**. This applies the
saved world origin and `orientation.controller_rotation` instead of anchoring
to the original start. It requires a calibration with that mounting rotation
and an unchanged Quest tracking session. This can expose constant frame bias
that relative conversion removes. Absolute errors also become absolute replay
target offsets; inspect the report before execution.

Outputs:

```text
DATASET/
  absolute_action_t/000000.npy    # [T,6]: calibrated Quest flange XYZ + rotation vector, Piper base
  original_action_t/000000.npy    # [T,6]: original FK interpolated at the SAME timestamps
  syncs/000000.npz                # pose_monotonic_ns, pose_frame=piper_base, source indices; no hand clock
  processed/episode_000000/
    piper.csv                    # calibrated Quest-derived flange, for inspection/replay format
    original_piper.csv            # aligned original FK
    errors.csv                   # position mm and SO(3) angular error degrees
    metadata.json                # metrics, calibration hash, latency and conversion mode
    comparison.png
```

Reports include RMSE, mean Euclidean/angular error, 95th percentile and maximum.
Postprocessing also creates `pose_dimensions.png/.pdf` (separate X/Y/Z and
roll/pitch/yaw overlays), `error_dimensions.png/.pdf` (signed axis differences,
rotation-error vector components and total errors), and `signed_errors.csv`.
Regenerate these plots from existing processed arrays without recalibration or
replay using:

```bash
python scripts/quest_piper/plot.py data_quest_piper --episode 0
```

Overlay Euler angles are unwrapped for display. Angular errors are computed on
SO(3); the XYZ angular components are a spatial rotation-error vector in Piper
base axes, not differences between Euler angles.

These are Quest-versus-FK agreement metrics, including both sensors, FK/model
error, mounting flex and timing effects. They are not an external measurement
of absolute robot accuracy. No independent calibration validation is implied
if the same episode was used to fit that calibration.

## 3. Plan, then execute replay

Planning checks the processed trajectory through the existing arm-only IK and
replay planner. By default it observes the connected arm's current joints and
controller limits but sends no actuator targets:

```bash
python scripts/quest_piper/replay.py data_quest_piper \
  --episode 0 --preposition --playback-speed 0.2
```

For offline planning without hardware, supply all six `--start-joints-deg`
values. These are a planning seed, not permission to skip measured start checks
on execution.

To enable and execute (including the validated approach if `--preposition`):

```bash
python scripts/quest_piper/replay.py data_quest_piper \
  --episode 0 --preposition --playback-speed 0.2 --execute
```

**`--execute` enables and moves the arm.** Configured software thresholds warn
instead of rejecting replay: Cartesian position/rotation jumps, joint steps and
speeds, start-pose mismatch, measured-to-target joint differences, joint/wrist
proximity margins and near-singular Jacobians. Speed thresholds are diagnostic,
not enforced caps in this mode. Missed deadlines shift the remaining schedule
instead of aborting or causing catch-up bursts; `schedule_delay_s` is logged and
removed from the source-time comparison. Connected planning uses the actual
controller-reported joint bounds rather than narrower software presets; offline
planning still has only preset bounds available.

Add **`--strict-limits`** to restore software threshold rejection and preset/controller
bound intersection. The underlying `replay_actions.py` stays strict by default;
this wrapper explicitly passes `--arm-only --warn-motion-limits --warn-margins`.
The settings are saved in the plan report. `--max-joint-speed-deg-s` (default 15)
sets the diagnostic threshold, or the enforced threshold in strict mode.
Valid data, converged IK, controller joint bounds, fresh feedback, and controller
fault checks remain required. Firmware limits are not changed. Smooth joint
commands and the existing enable/stop paths are retained. This does not check
table/tool collisions.
Execution reads and verifies the current arm even if an offline seed was given.
No hand is loaded or connected.

During execution the same accuracy viewer shows the original recorded Piper pose
in orange, current executed Piper FK in cyan, and the complete original FK path
in magenta. Position/orientation errors, RMSE, maxima, and red error segments all
compare the live executed Piper against the original recorded Piper at the
corresponding source time, accounting for playback speed and scheduling delays.
The approach is excluded from displayed statistics. IK status identifies
advisory versus strict validation; replay does not launch a
second competing IK solver. **Q/Esc**, closing the window, or Ctrl-C aborts via
the replay controller's existing stop path. `--no-view` suppresses the window.
No live Quest packets are needed for replay.

Outputs go to `DATASET/replays/<episode>_<timestamp>/`, or `--output-dir PATH`:

- `plan.npz` and `plan.json`: the existing replay planner's validated joint plan.
- `feedback.csv`: measured joints/FK, scheduled target, actual elapsed time,
  original-source time, approach/trajectory phase, and both error measures.
- `comparison.json`: replay-versus-original and replay-versus-scheduled-target
  metrics, excluding the approach; completion indicates whether replay returned normally.
- `comparison.png`: executed FK versus original trajectory and error traces.

Planning-only runs contain zero execution samples. Replay comparisons use
actual elapsed wall time, adjusted for playback speed, approach duration, and logged scheduling delays,
so servo lag remains visible. Feedback is sampled just before each command;
there is no final settled-pose measurement or automatic latency fitting.
An aborted replay retains available feedback and is marked incomplete.
The executed command path still comes from the calibrated Quest trajectory.
The primary viewer/report comparison is executed Piper versus recorded Piper;
scheduled-command tracking errors remain secondary diagnostics in saved reports.
