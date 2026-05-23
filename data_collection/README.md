# Midas/Piper Data Collection

This directory has a two-stage pipeline for diffusion-policy datasets:

1. `collect_raw_dataset.py` records raw episodes.
2. `postprocess_dataset.py` converts raw episodes into the trainer layout from
   `DIFFUSION_POLICY_DATASET_SPEC.md`.
3. `replay_episode.py` replays a raw episode on the Piper arm and Midas hand.

The collector is configured by editing defaults in:

```text
data_collection/collection/config.py
```

The main script also accepts a few device-selection overrides so you do not
need to edit config just to swap cameras:

```bash
python3 data_collection/collect_raw_dataset.py \
  --realsense-serial 123456789 \
  --teleop-camera /dev/video2
```

Useful discovery commands:

```bash
python3 data_collection/collect_raw_dataset.py --list-realsense
python3 data_collection/collect_raw_dataset.py --list-video-devices
python3 data_collection/collect_raw_dataset.py --list-input-devices
```

## Raw Collection

```bash
python3 data_collection/collect_raw_dataset.py
```

Sequence:

1. The script connects Piper, Midas hand, Paxini tactile, foot pedal, RealSense,
   and the teleop webcam.
2. RealSense and teleop preview windows stay live.
3. Piper teacher mode gates both teleop commands and recording.
4. Press `c` in a display window to capture teleop neutral calibration.
5. Press the foot pedal once.
6. After `READY_SECONDS = 1.0`, recording starts.
7. Press the foot pedal again to stop and save that episode.
8. The script returns to waiting for the next teacher-mode episode.
9. Use `Ctrl-C`, `q`, or `Esc` to fully stop the script.

Raw episodes are written under:

```text
data_midas/raw/episode_000000/
  camera.mp4
  camera_timestamps.csv
  midas.csv
  piper.csv
  tactile.npz
  metadata.json
```

If `RECORD_TELEOP_CAMERA = True`, raw episodes also contain:

```text
  teleop_camera.mp4
  teleop_timestamps.csv
```

Defaults:

- RealSense RGB: `640x480 @ 30Hz`
- RealSense selection: `CAMERA_SERIAL = None`; use `--realsense-serial` for a stable camera ID
- Teleop webcam selection: `TELEOP_CAMERA = 0`; use `--teleop-camera` for an OpenCV index or `/dev/v4l/by-id/...` path
- Teleop webcam overlay: shown
- Midas teleop commands: target `50Hz`, only while Piper teacher mode is active
- Midas measured positions: target `200Hz`
- Piper EEF pose: target `200Hz`
- Paxini tactile: target `60Hz`, saved compactly in `tactile.npz`
- Sync clock: host `monotonic_ns`

The collector owns the Midas motor serial port directly. Do not run a separate
`midas-hand-teleop` process at the same time.

## Code Layout

```text
data_collection/
  collect_raw_dataset.py
  collection/
    config.py
    episode.py
    shared_state.py
    realsense_worker.py
    teleop_worker.py
    midas_worker.py
    piper_worker.py
    tactile_worker.py
    runner.py
```

Each physical device is owned by one worker thread. The UI/runner owns foot
pedal polling, OpenCV display windows, keyboard neutral calibration, and episode
start/stop.

## Postprocessing

```bash
python3 data_collection/postprocess_dataset.py
```

The postprocessor writes:

```text
data_midas/
  camera/mp4_files/000000.mp4
  processed_encoder_t/000000.npy
  absolute_action_t/000000.npy
  syncs/000000.npz
```

Sync rows are keyed to RealSense frame timestamps. For each camera frame, the
nearest Midas hand sample and nearest Piper EEF sample are selected by
`monotonic_ns`.

The postprocessor uses measured Midas positions from `midas.csv`. It ignores
`tactile.npz` for now.

Piper raw `[x, y, z, roll, pitch, yaw]` is converted to trainer-ready
`[x, y, z, rotvec_x, rotvec_y, rotvec_z]` using fixed-axis XYZ Euler convention:

```text
R = Rz(yaw) * Ry(pitch) * Rx(roll)
```

## Replay

Replay the most recent raw episode at the default `30Hz`:

```bash
python3 data_collection/replay_episode.py
```

Replay a specific episode by index, name, or path:

```bash
python3 data_collection/replay_episode.py 42
python3 data_collection/replay_episode.py episode_000042
python3 data_collection/replay_episode.py data_midas/raw/episode_000042
```

Change replay frequency:

```bash
python3 data_collection/replay_episode.py --rate-hz 20
```

The replay script uses measured Midas positions from `midas.csv` and raw Piper
EEF `[x, y, z, roll, pitch, yaw]` from `piper.csv`. By default it moves slowly
to the first recorded arm/hand pose, settles briefly, then replays both streams
on a fixed-rate clock. It also supports `--hand-only`, `--arm-only`, and
`--no-preposition`.
