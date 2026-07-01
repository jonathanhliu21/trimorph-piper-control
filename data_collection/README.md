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

### Camera map

There are two independent camera "slots" that record video, plus RealSense
running in the background purely for depth. Match by slot purpose, not by
name — the physical device behind a slot can change over time:

| Slot     | Config prefix       | CLI flag                          | Physical device                                  | Saved to                                    |
|----------|----------------------|------------------------------------|------------------------------------------------------------|----------------------------------------------|
| task     | `TASK_CAMERA_*`      | (none — always depthai)            | Luxonis OAK-1 camera via the `depthai` SDK (see `test_oak1_camera.py`), auto-detected over USB — no `/dev/video` index to configure. Requires `pip install depthai`. | `camera.mp4` (the main training video) |
| teleop   | `TELEOP_CAMERA`      | `--teleop-camera`                  | Logitech UVC webcam. Primary MediaPipe hand-tracking input when `TELEOP_SOURCE="opencv"`, or a toggleable close-up "finger cam" (press `z`) when `TELEOP_SOURCE="realsense"`. | `teleop_camera.mp4` (only if `RECORD_TELEOP_CAMERA=True`) |
| aux      | `AUX_CAMERA_*`       | `--aux-camera`, `--no-aux-camera`  | Disabled by default. Unused unless you have a second, genuinely separate camera device beyond the task and teleop cameras above — **not** "the camera on the wrist" (that's the task camera). Also depthai-based (see `aux_camera_worker.py`). If you only have one OAK unit, don't enable this at the same time as the task camera — both would grab the same device. | `wrist_camera.mp4` (legacy file name, kept for dataset compatibility) |

RealSense is not a recording slot: it always runs in the background (when
`TELEOP_SOURCE="realsense"`) to supply depth/intrinsics for teleop wrist-pose
estimation, and is never written to video. Its resolution (`REALSENSE_WIDTH`/
`REALSENSE_HEIGHT`/`REALSENSE_FPS`) is independent of the task camera's —
check supported profiles with `--list-realsense` before changing it.

## Raw Collection

```bash
python3 data_collection/collect_raw_dataset.py
```

Sequence:

1. The script connects Piper, Midas hand, Paxini tactile, foot pedal, the task
   camera, RealSense (depth for teleop), and the teleop webcam.
2. Task view and teleop preview windows stay live (and a Logitech close-up
   window when `z` toggles it on).
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

- Task camera: always the OAK-1 (depthai), `TASK_CAMERA_WIDTH/HEIGHT/FPS = 1280x720 @ 30Hz`
- RealSense (teleop depth only, never recorded): `REALSENSE_WIDTH/HEIGHT/FPS = 640x480 @ 30Hz`; selection via `REALSENSE_SERIAL = None`, use `--realsense-serial` for a stable camera ID
- Teleop webcam selection: `TELEOP_CAMERA = <Logitech /dev/v4l/by-id path>`; use `--teleop-camera` for an OpenCV index or `/dev/v4l/by-id/...` path
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
    realsense_worker.py           # RealSense: teleop-depth feed only, never recorded
    depthai_task_camera_worker.py # task camera (OAK-1, always depthai) — records camera.mp4
    depthai_utils.py              # shared depthai pipeline/warmup helpers
    teleop_worker.py              # MediaPipe hand tracking (Logitech or RealSense input)
    aux_camera_worker.py          # optional separate camera slot, disabled by default (depthai)
    midas_worker.py
    piper_worker.py
    tactile_worker.py
    curobo_worker.py
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

Sync rows are keyed to task camera frame timestamps (`camera_timestamps.csv`).
For each camera frame, the nearest Midas hand sample and nearest Piper EEF
sample are selected by `monotonic_ns`.

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
