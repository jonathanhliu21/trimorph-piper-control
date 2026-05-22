# Midas/Piper Data Collection

This directory has a two-stage pipeline for diffusion-policy datasets:

1. `collect_raw_dataset.py` records raw episodes.
2. `postprocess_dataset.py` converts raw episodes into the trainer layout from
   `DIFFUSION_POLICY_DATASET_SPEC.md`.

Both scripts are configured by editing the variables near the top of the file.

## Raw Collection

```bash
python3 data_collection/collect_raw_dataset.py
```

Sequence:

1. The script connects Piper, Midas hand, foot pedal, and RealSense.
2. It waits for Piper teacher mode.
3. Press the foot pedal once.
4. After `READY_SECONDS = 1.0`, recording starts.
5. Press the foot pedal again to stop and save that episode.
6. The script returns to waiting for the next teacher-mode episode.
7. Use `Ctrl-C` to fully stop the script.

Raw episodes are written under:

```text
data_midas/raw/episode_000000/
  camera.mp4
  camera_timestamps.csv
  midas.csv
  piper.csv
  metadata.json
```

Defaults:

- RealSense RGB: `640x480 @ 30Hz`
- Midas hand positions: target `200Hz`, torque disabled
- Piper EEF pose: target `200Hz`
- Sync clock: host `monotonic_ns`

Note: if the hand teleop process owns the same Midas serial port, this collector
cannot also open the hand directly. The next integration should either have
teleop publish hand state for this collector, or move this capture loop into the
teleop process.

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

Piper raw `[x, y, z, roll, pitch, yaw]` is converted to trainer-ready
`[x, y, z, rotvec_x, rotvec_y, rotvec_z]` using fixed-axis XYZ Euler convention:

```text
R = Rz(yaw) * Ry(pitch) * Rx(roll)
```
