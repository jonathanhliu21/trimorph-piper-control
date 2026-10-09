# TRIMORPH + MIDAS/PiperX control

This repository contains hardware bring-up, contact-model testing, TRIMORPH-only
data collection, TRIMORPH-to-MIDAS retargeting and force feedback, episode
replay, and PiperX/OAK-1 collection tools.

Run commands from the repository root. Replace every example serial device with
the stable `/dev/serial/by-id/...` path for the corresponding hand.

## Safety and configuration

- TRIMORPH uses 13 XL330-M077-T motors with IDs `0..12`. During normal
  collection it is backdriven with torque disabled; force-feedback collection
  temporarily places selected leader joints in current mode.
- MIDAS is the commanded follower. Verify its calibration, limits, gains,
  current limits, motor order, and serial port before enabling torque.
- Motion tools refuse unreviewed YAML files. Check every calibration value and
  set `placeholders_unreviewed: false` only after review.
- Start motion tests at a conservative current limit with an unobstructed hand
  and access to power/E-stop controls. Options such as `--yes` and
  `--skip-target-safety` bypass safeguards and should only be used deliberately.

The two principal checked-in configurations are:

| Configuration | Used by |
| --- | --- |
| `scripts/trimorph_api_test/xl330_test_config.yaml` | TRIMORPH homing, angle recording, deterministic motion, current logging, and contact tests in `joint_test.py` |
| `config/midas_hand_config.yaml` | MIDAS contact UI, follower retargeting, replay onto MIDAS, and thumb calibration |

`data_collection/collect_hand_only.py` and direct TRIMORPH replay default to
`config/hand.yaml`. Create that local, calibrated runtime file from
`config/hand.example.yaml`, or pass `--hand-config` explicitly:

```bash
cp config/hand.example.yaml config/hand.yaml
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../trimorph_hand_api
```

The package installed from `trimorph_hand_api` is imported in this repository
as `midas_hand_api`. The core Python dependencies are NumPy, PyYAML (provided by
the hand API), and the hand API itself. Matplotlib is required for graph
generation. OpenCV and DepthAI are needed by the full camera collection
pipeline. Tkinter is required by the contact UIs and is normally supplied by
the system Python installation. The PiperX `pyAgxArm` package is installed
directly from the vendor's GitHub repository by `requirements.txt`; configure
SocketCAN before using PiperX.

Optional Paxini tactile operation also requires the Paxini support exposed by
`midas_hand_api` and a valid sensor serial port.

## 1. TRIMORPH finger bring-up and deterministic current tests

[`scripts/trimorph_api_test/joint_test.py`](scripts/trimorph_api_test/joint_test.py)
is the main bring-up tool. The global `--config` and `--test-finger` options go
before the subcommand. Finger choices are `thumb`, `finger-1`, `finger-2`,
`finger-3`, and `all`; non-homing commands default to `finger-1`.

Home one finger, or selected joints within that finger:

```bash
python3 scripts/trimorph_api_test/joint_test.py \
  --config scripts/trimorph_api_test/xl330_test_config.yaml \
  --test-finger finger-1 \
  home --current-threshold-ma 200 --velocity-rad-s 1.5

python3 scripts/trimorph_api_test/joint_test.py \
  --config scripts/trimorph_api_test/xl330_test_config.yaml \
  --test-finger thumb \
  home --joints 0 2
```

Omit `--test-finger` and `--joints` to home every configured motor. Homing
updates the selected motors' offsets while preserving the other saved offsets.

Read calibrated joint angles with torque disabled and write them to CSV:

```bash
python3 scripts/trimorph_api_test/joint_test.py \
  --config scripts/trimorph_api_test/xl330_test_config.yaml \
  --test-finger finger-1 \
  record --motors 4,5,6 --rate-hz 20 --duration-s 30 \
  --print-every 10 --output output/finger_1_angles.csv
```

Run a deterministic open-close-open path and capture graph-compatible current,
target/actual position, and target/actual velocity telemetry. `--duration-s` is
the duration of each leg, and `--trials` makes repeated trials available for
averaging:

```bash
python3 scripts/trimorph_api_test/joint_test.py \
  --config scripts/trimorph_api_test/xl330_test_config.yaml \
  --test-finger finger-1 \
  stream_move --motors 4,5,6 --rate-hz 50 --duration-s 2 \
  --settle-s 0.5 --trials 5 \
  --output-current output/finger_1_stream.csv
```

For discrete configured-profile moves instead, use:

```bash
python3 scripts/trimorph_api_test/joint_test.py \
  --config scripts/trimorph_api_test/xl330_test_config.yaml \
  --test-finger finger-1 \
  profile_move --motors 4,5,6 \
  --profile-velocity-rad-s 1.5 \
  --profile-acceleration-rad-s2 10 --trials 5 \
  --output-current output/finger_1_profile.csv
```

`profile_move_sequential` checks the configured motor mapping one motor at a
time. `measure_current`, `measure_position_error`, and `detect_contact` provide
continuous console measurements at selected targets. Run `python3
scripts/trimorph_api_test/joint_test.py COMMAND --help` for their complete
options. Existing `--output-current` files are protected unless `--force` is
passed.

## 2. MIDAS contact-model testing UI

[`scripts/trimorph_api_test/hand_ui.py`](scripts/trimorph_api_test/hand_ui.py)
opens a MIDAS controller and live contact visualizer:

```bash
python3 scripts/trimorph_api_test/hand_ui.py
```

The UI defaults to `config/midas_hand_config.yaml`; choose another reviewed
configuration in its **Config** field if needed. After connecting, use the
joint sliders or the finger/thumb open-close-open buttons. The UI displays
position/current-based contact state and exposes the motion-current threshold,
velocity threshold, settle/persistence timing, and contact feedback threshold.

To include Paxini fingertip contact sensing and a diagnostic log:

```bash
python3 scripts/trimorph_api_test/hand_ui.py \
  --add-tactile \
  --paxini-port /dev/serial/by-id/PAXINI \
  --tactile-debug-log output/paxini_contact.jsonl
```

`--paxini-port` or `--tactile-debug-log` also enables tactile sensing, so
`--add-tactile` is optional when either is supplied.

## 3. Graph generation from joint tests and replay

[`scripts/trimorph_api_test/gen_graphs_from_csvs.py`](scripts/trimorph_api_test/gen_graphs_from_csvs.py)
accepts the telemetry schema written by both `joint_test.py --output-current`
and `replay_episode.py --output-current`. The angle-only CSV produced by
`joint_test.py record` is not an input to this graph tool.

Generate an overview of current, commanded/measured position, and
commanded/measured velocity from a deterministic joint test:

```bash
python3 scripts/trimorph_api_test/gen_graphs_from_csvs.py \
  output/finger_1_stream.csv \
  --combine-mode average \
  --joints 4 5 6 \
  --titles "Finger 1 stream" \
  --output output/finger_1_stream.png
```

`--titles` requires exactly one title per input CSV. Compare joint-test and
replay telemetry directly with:

```bash
python3 scripts/trimorph_api_test/gen_graphs_from_csvs.py \
  output/finger_1_stream.csv output/replay_current.csv \
  --titles "Bench path" "Episode replay" \
  --overlay --plot-finger --combine-mode sequential \
  --joints 4 5 6 \
  --output output/bench_vs_replay.png
```

Use `--combine-mode average` for an interpolated mean and ±1 standard deviation
across trials, or `sequential` to concatenate trials in time. `--plot-finger`
uses finger-specific rows; without it the output is the three-panel overview.

## 4. Collect from TRIMORPH only

[`data_collection/collect_hand_only.py`](data_collection/collect_hand_only.py)
can record the backdriven TRIMORPH without launching PiperX. The wrist camera
is optional:

```bash
python3 data_collection/collect_hand_only.py \
  --data-collection \
  --dataset-root data_trimorph \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --active-motors 0 1 2 3 4 5 6 7 8 9 10 11 12 \
  --warmup-time 5 --episode-time 30
```

It warms up, records one fixed-duration episode, and exits. A hand-only episode
contains populated `leader_hand.csv` and `metadata.json`, plus empty
`camera_timestamps.csv` and `piper.csv` bookkeeping files. `follower.csv` is
written only when a follower is configured, and no camera video is created.
The older filename `proprioception.csv` is still accepted by replay for
compatibility.

Add `--add-camera` to start the DepthAI/OAK wrist camera, display its live feed,
and save `camera.mp4` plus populated `camera_timestamps.csv` and camera metadata
in the same format as the full collector. Press `q` or Escape in the camera
window to stop through the normal episode-save path. Camera brightness and
anti-banding defaults are configured by `TASK_CAMERA_BRIGHTNESS` and
`TASK_CAMERA_ANTI_BANDING_HZ` in `data_collection/collection/config.py`; the
default anti-banding mode is `60` Hz.
Set `TASK_CAMERA_FLIP = "180"` in the same config file to rotate the saved and
previewed image by 180 degrees.

## 5. TRIMORPH leader to MIDAS follower with force feedback

Provide both follower options to enable retargeting. `--retarget-only` runs
until Ctrl-C without saving an episode; omit it (or use `--data-collection`) to
record one fixed-duration episode:

```bash
python3 data_collection/collect_hand_only.py \
  --retarget-only \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --follower-config config/midas_hand_config.yaml \
  --follower-port /dev/serial/by-id/MIDAS \
  --thumb-model thumb_retarget_model.json \
  --show-ui
```

TRIMORPH positions are scaled/offset, the learned thumb model replaces MIDAS
joints 2-3, and targets are clipped and smoothed before being sent to MIDAS.
The follower's detected contact error is sent back to selected TRIMORPH joints
as capped current-mode force feedback. `--show-ui` requires a follower. Add
`--add-tactile --paxini-port /dev/serial/by-id/PAXINI` to combine Paxini
fingertip contacts with force feedback, or use `--tactile-debug-log PATH` for
headless tactile diagnostics.

## 6. Episode replay and current logging

Replay the latest episode on TRIMORPH and PiperX (or supply an episode number,
`episode_XXXXXX`, or a directory path):

```bash
python3 data_collection/replay_episode.py \
  --dataset-root data_trimorph \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH
```

Replay just episode 42's hand stream and record graph-compatible current
telemetry at the replay logger's fixed 10 Hz sampling rate:

```bash
python3 data_collection/replay_episode.py 42 \
  --dataset-root data_trimorph \
  --hand-only \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --rate-hz 30 --preposition-seconds 3 --settle-seconds 1 \
  --output-current output/episode_42_replay_current.csv
```

To replay the recorded TRIMORPH stream as retargeted commands on MIDAS:

```bash
python3 data_collection/replay_episode.py 42 \
  --dataset-root data_trimorph \
  --hand-only --use-follower \
  --hand-config config/midas_hand_config.yaml \
  --hand-port /dev/serial/by-id/MIDAS \
  --thumb-model thumb_retarget_model.json \
  --output-current output/episode_42_midas_current.csv
```

Useful switches are `--no-preposition`, `--arm-only`, and `--no-piper-enable`.
`--hand-only` and `--arm-only` are mutually exclusive in practice, and current
logging cannot be used with `--arm-only`.

### Alternating manual replay

[`scripts/replay_corl_workshop/manual_replay.py`](scripts/replay_corl_workshop/manual_replay.py)
replays separate Piper and TRIMORPH trajectories one at a time. Create a text
file whose non-comment lines alternate from Piper to TRIMORPH; relative paths
are resolved from the text file's directory:

```text
# workshop_sequence.txt
trajectories/piper_pick.csv
trajectories/trimorph_close.csv
trajectories/piper_place.csv
trajectories/trimorph_open.csv
```

Then run:

```bash
python3 scripts/replay_corl_workshop/manual_replay.py workshop_sequence.txt \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --foot-pedal-device /dev/input/by-id/FOOT_PEDAL
```

Both devices are checked before motion. The first fresh pedal press moves the
hand to the initial joint-angle constant defined in the script. After that move
settles, a second fresh press starts the first Piper trajectory. After each CSV
finishes, press and release the foot pedal to advance; a press made during
motion is discarded. The inactive controller stays enabled at its last target,
and no trajectory commands are sent to it until its next turn. Use
`--list-input-devices` to find the pedal event device and `--help` for
replay-rate, preposition, hardware, and motion-profile options.

## 7. Thumb IK/retarget calibration

[`data_collection/retarget_thumb.py`](data_collection/retarget_thumb.py)
collects corresponding TRIMORPH and MIDAS thumb poses for the joints 2-3
retarget map. TRIMORPH stays torque-disabled while MIDAS moves through a
deterministic, joint-limit-aware calibration set:

```bash
python3 data_collection/retarget_thumb.py \
  --trimorph-config config/hand.yaml \
  --trimorph-port /dev/serial/by-id/TRIMORPH \
  --midas-config config/midas_hand_config.yaml \
  --midas-port /dev/serial/by-id/MIDAS \
  --output thumb_tip_matching.csv \
  --num-points 40 --seed 7 --limit-margin 0.05 \
  --max-step 0.04 --command-rate 50 \
  --settle-time 0.5 --samples 10 --sample-rate 100
```

At each prompt, hold the backdriven TRIMORPH thumb at the pose corresponding to
the displayed MIDAS pose and press Enter. The CSV is checkpointed after every
point. Then fit the affine retarget model used by collection and follower
replay:

```bash
python3 data_collection/fit_model_to_endpoints.py \
  --input thumb_tip_matching.csv \
  --output thumb_retarget_model.json
```

Use `--no-configure-midas` only when the required MIDAS operating mode and gain
registers have already been configured correctly.

## Important shared modules

| Module | Role |
| --- | --- |
| `data_collection/collection/contact_detection.py` | Shared current/velocity/position-error state machine and thresholds used directly by `joint_test.py` and `hand_ui.py`, and by the MIDAS follower worker during force-feedback retargeting |
| `data_collection/collection/follower_hand_worker.py` | TRIMORPH-to-MIDAS scaling, thumb-model application, smoothing, follower contact detection, and follower telemetry |
| `data_collection/collection/leader_hand_worker.py` | Torque-disabled leader reads plus capped current-mode force feedback on TRIMORPH |
| `data_collection/collection/thumb_retargeting.py` | Loads and applies the learned joints 2-3 thumb retarget model |
| `data_collection/collection/hand_common.py` | Loads narrowed hand configs and verifies connected Dynamixel models |
| `data_collection/collection/tactile_contact_detection.py` | Paxini moving-average contact detection used by the tactile UI/worker paths |

Changing contact thresholds in `contact_detection.py` affects both standalone
contact test scripts and the force-feedback follower path; validate changes on
low-current hardware tests.

## Full PiperX/OAK-1 collection and dataset conversion

The full collector remains available when synchronized camera, PiperX, and hand
data are required:

```bash
python3 data_collection/collect.py \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --foot-pedal-device /dev/input/by-id/FOOT_PEDAL
```

Its raw episode contains `camera.mp4`, `camera_timestamps.csv`,
`leader_hand.csv`, optional `follower.csv`, `piper.csv`, and `metadata.json`.
Convert raw episodes with:

```bash
python3 data_collection/postprocess_dataset.py data_trimorph
```

See `data_collection/DATASET_SPEC.md` for the processed format and
`data_collection/README.md` for additional implementation details.

## Verification

```bash
python3 -m compileall data_collection scripts tests
PYTHONPATH=data_collection python3 -m unittest discover -s tests
```

These checks do not exercise hardware. Before a full run, perform a low-current
open/mid/fist smoke test and compare the leader target, follower command, and
follower measurement for every motor.
