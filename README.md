# TRIMORPH + PiperX Data Collection

This repository is the TRIMORPH-specific successor to `midas-piper-control`.
It intentionally supports one collection stack:

```text
OAK-1 wrist camera  -> camera.mp4
PiperX drag-teach   -> piper.csv
Backdriven TRIMORPH -> proprioception.csv
Follower MIDAS      -> follower.csv (optional)
```

MediaPipe retargeting, Logitech/RealSense teleoperation, auxiliary cameras,
cuRobo, MIDAS/XM335 fallbacks, and duplicate collection runners were removed.
The Python import provided by `trimorph_hand_api` is still named
`midas_hand_api`; that upstream package name is the only intentional MIDAS
name remaining in runtime code.

## Safety assumptions

- The backdriven TRIMORPH uses XL330-M077-T motors with IDs `0..12`.
- Its YAML file describes calibrated coordinates in radians.
- It is torque-disabled during collection and is never commanded.
- If configured, the follower uses a separate serial port and a compatible
  motor ordering; it is model-verified before configuration or motion commands.
- Placeholder example YAML files are not runtime defaults. Copy them and
  replace every calibration value before connecting hardware.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../trimorph_hand_api
```

Install the PiperX `pyAgxArm` package separately using the vendor-provided
instructions. Configure the SocketCAN interface before collection.

Create calibrated hand configurations:

```bash
cp config/hand.example.yaml config/hand.yaml
```

The examples contain deliberately narrow placeholder limits. Do not operate
the follower until offsets, signs, limits, gains, and current limits have been
reviewed for the physical mechanisms.

## Collect

Edit the hardware defaults in `data_collection/collection/config.py`, or pass
the device-specific values explicitly:

```bash
python3 data_collection/collect.py \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --foot-pedal-device /dev/input/by-id/FOOT_PEDAL
```

To mirror the primary hand onto a separate MIDAS-family device, add
`--follower-config` and
`--follower-port`; omitting `--follower-config` disables that worker.

The Piper worker is passive. Move the arm with its built-in drag-teach mode.
Press and release the pedal to start an episode, then press and release it to
save. Hold it for two seconds while recording to discard the episode. Press
`q` in the OAK preview to stop.

Follower smoothing defaults to `alpha=0.4` and a `0.15 rad` maximum step. For
literal one-sample forwarding set `follower_interpolation_alpha=1.0` and
`follower_max_step_rad=0.0` after confirming calibration and limits.
`follower_joint_command_scales` provides an additional motor-ID keyed scale;
all motor scales default to `1.0` and multiply `follower_command_scale`.

## Dataset

Each raw episode contains:

- `camera.mp4` and `camera_timestamps.csv`: wrist-mounted OAK policy input;
- `proprioception.csv`: measured positions from the backdriven TRIMORPH;
- `follower.csv`: optional follower measurements;
- `piper.csv`: PiperX end-effector state;
- `metadata.json`: timing, effective hand model/baud/current settings, config
  paths, and SHA-256 fingerprints.

Convert the raw data with:

```bash
python3 data_collection/postprocess_dataset.py data_trimorph
```

See `data_collection/DATASET_SPEC.md` for the processed layout.

Replay an episode on the same hand and PiperX with:

```bash
python3 data_collection/replay_episode.py 42 \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH
```

Replay interpolates both streams on one episode-time clock. The hand uses a
3.0 rad/s velocity and 50.0 rad/s² acceleration motion profile.

## Verification

```bash
python3 -m compileall data_collection scripts tests
python3 -m unittest discover -s tests
```

Compilation and unit tests do not exercise hardware. Before real collection,
perform a low-current open/mid/fist smoke test while comparing the primary
target, follower command, and follower measurement for every motor.
