# Trimorph policy deployment

This ports the MIDAS predictor/protocol/executor into this repository. The GPU
machine runs the shared diffusion-policy predictor. The robot machine runs the
Trimorph/Piper executor and the same OAK camera backend/settings as collection.
It does not import the MIDAS controller repository.

## Install

On the robot machine, in its control environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ../trimorph-hand-api
python -c 'import midas_hand_api; print(midas_hand_api.__file__)'
```

The Trimorph fork retains the `midas-hand` distribution name, `midas_hand_api`
module, and `MidasHand` class. The executor defaults to the sibling
`../trimorph-hand-api` source directory and checks the imported module location.
Use `--hand-api-repo` if it lives elsewhere. Keep the GPU training environment
separate from the hardware dependencies.

On the GPU machine, install the dependencies for your `diffusion-policies`
checkout. Run from this repository's root (the checkout can live on each machine
at a different path):

```bash
python inference/predictor/predictor.py \
  --policy-repo /path/to/diffusion-policies \
  --checkpoint /path/to/trimi/checkpoints/latest.ckpt \
  --device cuda:0
```

No stale machine-specific checkpoint is selected automatically. The predictor
loads the checkpoint's normalizer and sampling settings. To experiment with
fewer denoising passes, pass `--num-inference-steps 8` and evaluate policy quality.

## Test the connection without hardware

With the predictor running:

```bash
python inference/executor/executor.py \
  --predictor GPU_HOST --fake --max-cycles 20
```

Fake mode uses synthetic images/state and does not connect to the hand, Piper,
or camera. The real GPU model still runs. Unit tests need no GPU or devices:

```bash
python -m unittest discover -s tests -p 'test_*inference.py'
```

## Execute on Trimorph and Piper

```bash
python inference/executor/executor.py --predictor GPU_HOST --hz 15
```

The hand defaults are the bench TriMi: `../trimorph-hand-api`,
`config/xl330_test_config.yaml`, and the FTDI `FTBEO9P0` serial port. That
adapter is shared with the glove, so confirm TriMi is on it, or pass
`--hand-api-repo`, `--hand-config`, and `--hand-port`.

This enables and commands the actual hand and arm. Before the first observation
the arm ramps in joint space (10 deg/s, after an Enter prompt) to a named start
from `PIPER_START_JOINTS_DEG` in `data_collection/collection/config.py`:
`--start-home home_4c` by default (logged start #1, a tested offset of `home_4b`),
`home_4d` (tested start #6), `home_4b` (median start of episodes 95-220 of
`data_trimorph_hand_mocap_quest_4`), `home_4a` (its episodes 0-94),
`home_cube1` (median start of `data_trimorph_hand_mocap_quest_5`), or `quest_0`.
Every start reached (startup and pedal resets) is appended to
`output/executor_starts.jsonl` with an id, the commanded and measured joints, and
the measured flange pose; `--start-id N` repeats start N exactly (joints and hand
start). `--start-jitter-mm` / `--start-jitter-deg` (default 0) add a random flange
offset around the chosen start, solved by IK with replay's joint checks and logged
like any other start.
Use `--start-joints-deg J1..J6` for another start, `--start-speed-deg-s` to change
the speed (max 30), or `--no-start-move` to start wherever the arm and hand are.
After the arm, the hand moves (at the configured hand profile velocity) to the
same start's median demo hand pose, `HAND_START_POSITIONS_RAD`; a hand that is not
within 0.05 rad after 5 s only prints a warning. `--no-hand-start` moves only the arm. `--show`
opens a live wrist-camera window (flipped like the policy input); in synchronous
mode it freezes while waiting for the predictor. q/Esc stops the executor.
The arm is driven in joint space by default (`--arm-control joint`,
`inference/executor/joint_control.py`): each policy pose is solved by local IK
from the previous joint target and sent with `move_j`. The IK is exact except
near the singularity (Jacobian smallest singular value below
`--ik-damping-sigma`, default 0.03; 0.087 at `home_cube1`), where it accepts a
few mm of pose error instead of large J2/J3 jumps; each step is scaled down to
the joint speed caps (`--max-joint-speed-deg-s`, default 45 45 90 60 45 60), and
joints stay 3 deg inside the SDK limits with |J5| >= 10 deg. `--arm-control
pose` restores the firmware Cartesian `move_p`, which oscillated in J2/J3 near
the edge of reach. Joint targets are not sent as one jump per action: each is
reached over the action period by evenly spaced, linearly interpolated `move_j`
commands at about `--command-hz` (default 60: 4 per action at `--hz 15`), which
removes the per-action stop-and-go of a 15 Hz stream at the cost of about half
an action period (~33 ms) of extra lag. `--command-hz 15` sends one per action.
The predictor starts every chunk exactly at the arm pose in the observation
state (checked on run `20261104_113450`: chunk start moves 1.00x with the
observed pose). With the measured pose, the arm's tracking lag or swing at
observation time is therefore copied into each chunk. By default
(`--state-anchor command`) the observation carries the last commanded pose
instead (in joint mode, FK of the joint target sent), so each chunk continues
the commanded path; if the arm is more than `--anchor-max-mm` (50) or
`--anchor-max-deg` (20) away from it, the measured pose is sent and a warning
printed. `--state-anchor measured` restores the previous behavior. The hand
state is always measured.
Each run is logged to `output/executor_runs/<timestamp>/` (`--log-root`, or
`--no-log`): `piper.csv` polls the Piper at 200 Hz (joints, motor velocity and
current, firmware flange pose, `motion_status`), `commands.csv` has every arm
target sent (raw policy and EMA-filtered, XYZ + Euler), `chunks.npz` the raw
predictor chunks with the observed state, and `events.csv` the observe/chunk
timing. Everything is held in memory and written when the executor exits,
then `plot_run.py` saves `joints.png` and `eef.png` (position, velocity, and
acceleration of each joint and flange axis, measured vs commanded, with
predictor waits shaded) and `spectrum.png` into the same folder, and prints
per-band velocity RMS: measured well above commanded means the arm adds the
shake; equal means it is in the commands. Replot or zoom with
`python inference/executor/plot_run.py RUN_DIR --start 10 --end 16`.
`--traj-view [DATASET]` opens a 3D window with every demonstrated flange
trajectory of `DATASET` (default `data_trimorph_hand_mocap_quest_4`), the live
EEF pose and trail, the chunk being executed, and the distance to the nearest
demo point; `python inference/executor/trajectory_view.py DATASET` shows the
trajectories alone. Hand setup follows the repository's
replay path: mode 3 position control, hold measured position before enabling
torque, configured velocity/acceleration profiles, then nonblocking position
writes. It does not start the collection force-feedback workers. On exit it
shuts down the hand and disconnects Piper; it does not retract/home the arm.

Piper firmware/interface/channel/installation/payload and OAK image size, FPS,
flip, manual exposure/ISO, and white-balance defaults come from
`data_collection/collection/config.py`. Override them through `--help` options
when collection used different settings. The hand API uses the YAML baudrate,
calibration, gains, and limits. It verifies the connected motor models.

## Coordinate contract

- State and absolute actions are `[XYZ_m, rotvec_rad, hand_rad]`, shape `[19]`.
- Hand elements are exactly motor IDs `0..12`, in leader coordinates. Use the
  same signs, home offsets, and zero calibration as collection. There is no
  follower scaling, thumb retargeting, or passive-joint/16-DOF expansion.
- `read_pos()` supplies calibrated hand positions. `set_positions()` applies
  hardware signs/home offsets once and clips to configured joint limits.
- Piper feedback Euler xyz angles are converted to rotation vectors before
  inference; predicted rotation vectors are converted back to Euler xyz for
  Piper commands. They must never be passed through as interchangeable angles.
- Camera frames are flipped like collection and converted from OAK BGR to RGB.
- Quest-derived training poses should be processed with the actual
  `--piper-start-pose` so orientation-dependent frame conversion agrees with
  execution. The inference protocol does not carry pose-frame metadata.
- `--hz` must match the synchronized training sample rate after downsampling.
  A 30 FPS dataset with stride 2 uses 15 Hz, not the raw hand measurement rate.

## Timing and limitations

Before playback, the executor checks the entire action chunk for Euclidean XYZ
jumps greater than `MAX_EEF_JUMP_M = 0.15` meters, defined in
`inference/executor/executor.py`. It checks the observation-to-first-action jump
and every consecutive action pair. Any violation prints a warning and stops
the executor without applying any hand or arm actions from that chunk. This
check does not enforce workspace, table clearance, joint limits, or rotation limits.

This port retains synchronous request/reply execution. It captures an
observation, waits for the predictor, and then executes the returned chunk.
Inference therefore remains a pause between chunks by default. Enable overlapping
inference on the executor with:

```bash
python inference/executor/executor.py --predictor GPU_HOST --hz 15 \
  --async-inference --inference-latency-ms 70
```

The predictor command and protocol stay the same. To test the link without
hardware, add `--fake --max-cycles 20 --no-log` to this command.

Use `--execute-steps K` to execute at most the first K actions of each returned
chunk, in either synchronous or async mode. The default `0` uses the entire
returned chunk. If fewer than K actions arrive, use the available count. Raw
chunks are still logged and validated in full, while the trajectory view shows
the chosen prefix.

In async mode, the executor forecasts the current chunk's finish from its
remaining **chosen** actions and `--hz`, and captures the next observation
`--inference-latency-ms` before that finish (default **70 ms**). JPEG encoding,
network I/O, and prediction run in the background while playback continues.
For N chosen actions at frequency f, nominal execution takes `N / f` seconds,
so the next request starts about `max(0, N / f - latency_seconds)` after playback
starts. The deadline is recalculated after every action from its actual send
time and the number of chosen actions remaining. For example, with
`--hz 15 --execute-steps 8`, an available 8-action prefix takes about 533 ms and
the next request starts about 463 ms into playback with the 70 ms lead.
`--command-hz` controls interpolation within each action; it does not multiply
the number of policy actions or change this duration calculation.
The estimate should cover the full request/reply latency, including inference;
increase it if replies routinely arrive late. It is a fixed tunable estimate,
not an automatically measured delay. A value of 0 requests at the chunk end.

When a reply is ready, it replaces the previous chunk **at the next policy action
boundary**: the current action's interpolated arm commands finish, then all
unexecuted old actions are discarded. The first action of the new chunk plays
next, with EMA history retained. For example, a 10-action chunk at 100 Hz lasts
100 ms: the next request starts around 30 ms with a 70 ms estimate. If the reply
arrives at 35 ms, it takes over at 40 ms and drops the six remaining old actions.

There is at most one request in flight. A chunk shorter than the lead time starts
its next request after issuing its first action. The first chunk requires an
initial wait; if a later reply is late, the chosen prefix finishes and the robot
holds its final target until the reply arrives, without playing the excluded
suffix. Playback never speeds up to catch
up. Hardware reads, writes, and GUI updates stay on the control thread; async
camera observations use the newest available frame without blocking for a new
one. A pedal reset discards the old tail and any in-flight reply, drains that
reply, and takes a fresh observation after resuming. `--max-cycles` counts
requests, including any reply discarded on reset.

The predictor still anchors absolute actions to the state captured **at request
time**. The robot can move during prediction, so the start of a new chunk can
point back toward that earlier state. This implementation does not extrapolate
the observation, rebase the returned chunk, or skip its first actions. The
existing whole-chunk jump check also checks the last commanded action against
the replacement's first action in async mode.

Each cycle logs observation time, full predictor `infer` time, request/reply
`rtt`, execution time, and action deadline overruns. `rtt` includes prediction;
do not add the two values. `observe + rtt` is the approximate inter-chunk overhead
on top of the scheduled action interval in synchronous mode. Predictor timing covers CPU output
conversion as well as model execution. Socket and camera timeouts bound waits;
invalid/nonfinite outputs or mismatched sequence numbers abort execution.

Cycle logs include `planned`, `executed`, and `dropped` action counts; `dropped`
includes both the excluded suffix and any early-replaced actions. `execute` is
measured wall-clock playback time, which can differ from nominal `N / f` due to
early replacement, IK/IO overhead, or deadline overruns. Async run events include
`chunk_start`, `wait_start`/`wait_end`, and replies discarded on a pedal reset;
`chunk_received` records when the controller takes the reply at an action
boundary, while `rtt` measures the actual socket exchange. Async plots shade only
playback waits, not inference that overlaps execution. No hardware performance
or policy success is established by unit tests or the synthetic link test.
