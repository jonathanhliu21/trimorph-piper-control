# TRIMORPH + PiperX Data Collection

This repository is the TRIMORPH-specific successor to `midas-piper-control`.
It intentionally supports one collection stack:

```text
OAK-1 wrist camera  -> camera.mp4
PiperX drag-teach   -> piper.csv
TRIMORPH leader     -> trimorph_leader.csv -> follower target
TRIMORPH follower   -> trimorph.csv (measured policy state)
```

MediaPipe retargeting, Logitech/RealSense teleoperation, auxiliary cameras,
cuRobo, MIDAS/XM335 fallbacks, and duplicate collection runners were removed.
The Python import provided by `trimorph_hand_api` is still named
`midas_hand_api`; that upstream package name is the only intentional MIDAS
name remaining in runtime code.

## Safety assumptions

- Both hands use XL330-M077-T motors with IDs `0..12`.
- Leader and follower use different, explicit serial ports.
- Their YAML files describe calibrated coordinates with matching joint order,
  signs, zeros, and radians convention.
- The leader is torque-disabled and is never commanded.
- The follower is model-verified before configuration or motion commands.
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
cp config/leader_hand.example.yaml config/leader_hand.yaml
cp config/follower_hand.example.yaml config/follower_hand.yaml
```

The examples contain deliberately narrow placeholder limits. Do not operate
the follower until offsets, signs, limits, gains, and current limits have been
reviewed for the physical mechanisms.

## Collect

Edit the hardware defaults in `data_collection/collection/config.py`, or pass
the device-specific values explicitly:

```bash
python3 data_collection/collect.py \
  --leader-config config/leader_hand.yaml \
  --leader-port /dev/serial/by-id/LEADER \
  --follower-config config/follower_hand.yaml \
  --follower-port /dev/serial/by-id/FOLLOWER \
  --foot-pedal-device /dev/input/by-id/FOOT_PEDAL
```

The Piper worker is passive. Move the arm with its built-in drag-teach mode.
Press and release the pedal to start an episode, then press and release it to
save. Hold it for two seconds while recording to discard the episode. Press
`q` in the OAK preview to stop.

Follower smoothing defaults to `alpha=0.4` and a `0.15 rad` maximum step. For
literal one-sample forwarding set `follower_interpolation_alpha=1.0` and
`follower_max_step_rad=0.0` after confirming calibration and limits.

## Dataset

Each raw episode contains:

- `camera.mp4` and `camera_timestamps.csv`: wrist-mounted OAK policy input;
- `trimorph_leader.csv`: leader encoder targets for diagnostics;
- `trimorph.csv`: follower measured positions used for training;
- `piper.csv`: PiperX end-effector state;
- `metadata.json`: timing, effective hand model/baud/current settings, config
  paths, and SHA-256 fingerprints.

Convert the raw data with:

```bash
python3 data_collection/postprocess_dataset.py data_trimorph
```

See `data_collection/DATASET_SPEC.md` for the processed layout.

## Verification

```bash
python3 -m compileall data_collection scripts tests
python3 -m unittest discover -s tests
```

Compilation and unit tests do not exercise hardware. Before real collection,
perform a low-current open/mid/fist smoke test while comparing leader target,
follower command, and follower measurement for every motor.
