# Diffusion Policy Dataset Spec For TRIMORPH/Piper Collection

This document describes the exact dataset format expected by the diffusion
policy trainer in `/home/alvinz/alvinz/diffusion-policies` for the TRIMORPH hand
plus Piper arm setup.

The trainer config is:

```text
diffusion_policy/config/train_unet_image_trimorph_hand_cond_dinov2_policy.yaml
diffusion_policy/config/tasks/trimorph_image_hand_cond.yaml
```

The dataset class is:

```text
shared/env/trimorph/trimorph_image_dataset.py
```

## High-Level Contract

Each training episode must provide synchronized streams for:

- RGB camera frames.
- Piper EEF pose, 6 values per sample.
- TRIMORPH hand active joint positions, 13 values per sample.

The trainer stores an absolute state per synchronized policy timestep:

```text
absolute_state[t] = concat(eef_pose_6[t], trimorph_hand_13[t])
```

Then the trainer converts each sampled training window into relative training
targets:

```text
eef_target[t]  = SE3_relative(eef_pose_6[t], eef_pose_6[window_start])
hand_target[t] = trimorph_hand_13[t] - trimorph_hand_13[window_start]
action[t]      = concat(eef_target[t], hand_target[t])
```

Do not save normalized actions. Do not save pre-windowed deltas. Save absolute
EEF pose and absolute hand position streams; the diffusion-policy dataset loader
does the relative conversion and normalization during training.

## Required Dataset Root Layout

Create one dataset root with this exact structure:

```text
<dataset_root>/
  camera/
    mp4_files/
      000000.mp4
      000001.mp4
      ...
  processed_encoder_t/
    000000.npy
    000001.npy
    ...
  absolute_action_t/
    000000.npy
    000001.npy
    ...
  syncs/
    000000.npz
    000001.npz
    ...
```

The loader also accepts the same structure nested under `<dataset_root>/data/`,
but prefer the direct structure above unless there is a reason to keep raw data
separate.

The four folders must contain the same number of files. Files are matched by
lexicographic sort, so use zero-padded episode names. For example, these four
files define one episode:

```text
camera/mp4_files/000042.mp4
processed_encoder_t/000042.npy
absolute_action_t/000042.npy
syncs/000042.npz
```

## Camera Video Requirements

`camera/mp4_files/*.mp4` contains one RGB video per episode.

The diffusion task config declares camera observations as:

```yaml
camera_0:
  shape: [3, 240, 320]
  type: rgb
```

That is channels-first `[C, H, W]`, so the dataset frame size is:

```text
height = 240
width  = 320
```

The loader reads video frames with OpenCV, converts BGR to RGB, and resizes the
full frame to `320x240` using `cv2.INTER_AREA` if needed. It does not crop in
the dataset loader. If the raw video is `640x480`, it will be downsampled to
`320x240`.

The DINOv2 encoder later applies its own training-time preprocessing:

```yaml
resize_shape: [240, 240]
crop_shape: [224, 224]
random_crop: True
```

The collection code only needs to save readable MP4s. Keep exposure, camera
pose, and camera intrinsics stable across episodes if possible.

## TRIMORPH Hand Array Requirements

`processed_encoder_t/*.npy` must be a NumPy array:

```text
shape: [T_hand, 13]
dtype: float32 preferred; float64 is acceptable but will be cast to float32
meaning: absolute current TRIMORPH hand positions
```

These are absolute positions in the command/state scale that should be used for
robot execution. Do not normalize these values to `[0, 1]` or `[-1, 1]`.

Use a fixed 13-DOF ordering for every episode. The expected ordering should
match the TRIMORPH hand API motor order used for command/replay. Based on this repo:

```text
thumb:  motor ids 0, 1, 2, 3
index:  motor ids 4, 5, 6
middle: motor ids 7, 8, 9
ring:   motor ids 10, 11, 12
```

Recommended 13D order:

```text
[pos_0, pos_1, pos_2, pos_3,
 pos_4, pos_5, pos_6,
 pos_7, pos_8, pos_9,
 pos_10, pos_11, pos_12]
```

If the API returns `hand.motor_ids` in a different order, either save the array
in `hand.motor_ids` order and document that order in metadata, or remap it to
the canonical order above before writing `processed_encoder_t`.

The trainer default is:

```yaml
encoder_window: 0
```

So the diffusion-policy dataloader does not smooth the hand signal. If filtering
is needed, do it in this data-collection/postprocessing repo before writing
`processed_encoder_t`. Keep the filter documented: sample rate, cutoff, causal
vs zero-phase, and whether the same filtered signal will be available during
deployment.

## Piper EEF Array Requirements

`absolute_action_t/*.npy` must be a NumPy array:

```text
shape: [T_eef, 6]
dtype: float32 preferred; float64 is acceptable but will be cast to float32
meaning: absolute Piper EEF pose
```

Expected EEF pose format for diffusion training:

```text
[x, y, z, rx, ry, rz]
```

where:

- `x, y, z` are position in meters.
- `rx, ry, rz` are rotation-vector components in radians.

Important: the current Piper examples in this repo record flange pose as:

```text
[x, y, z, roll, pitch, yaw]
```

The diffusion-policy SE(3) helper currently interprets the last three values as
a rotation vector, not Euler roll/pitch/yaw. Therefore the data curation code
must either:

1. Convert Piper roll/pitch/yaw to rotation vector before writing
   `absolute_action_t/*.npy`, or
2. Update the diffusion-policy SE(3) helper and TRIMORPH inference code to use the
   same Euler convention.

The intended path is option 1. Save `absolute_action_t` as `[x, y, z, rotvec]`.
The inference wrapper will return absolute EEF targets in the same convention,
so downstream robot execution must convert rotvec back to whatever Piper command
API expects if the Piper API expects roll/pitch/yaw.

## Sync File Requirements

`syncs/*.npz` must contain three integer arrays:

```text
frame_idx
encoder_idx
pose_idx
```

All three arrays must have the same length `N`, where `N` is the number of
synchronized policy timesteps in the episode.

At synchronized timestep `k`:

```text
camera frame       = video_frames[frame_idx[k]]
TRIMORPH hand state   = processed_encoder_t[encoder_idx[k]]
Piper EEF pose     = absolute_action_t[pose_idx[k]]
```

Use zero-based indices. Keep indices in range. The current loader clips indices
to valid ranges, but the collection pipeline should treat out-of-range indices
as a bug and fail validation rather than relying on clipping.

Each sync row should represent one policy/control timestep. The training config
uses:

```yaml
horizon: 16
num_obs_steps: 1
num_action_steps: 8
num_inference_steps: 16
```

The trainer can pad near episode boundaries, but useful episodes should contain
many more than 16 synchronized rows.

## What The Policy Sees And Predicts

The TRIMORPH policy observation metadata is:

```yaml
obs:
  camera_0:
    shape: [3, 240, 320]
    type: rgb
  hand_pose_abs:
    shape: [13]
    type: low_dim
action:
  shape: [19]
```

The policy condition is:

```text
image + current absolute hand pose
```

The policy does not condition on absolute EEF pose. The EEF pose is present only
inside the action/state stream so the training target can be a relative EEF
trajectory.

During inference, the wrapper expects:

```text
robot_state = concat(current_eef_abs_6, current_hand_abs_13)
shape: [19]
```

It feeds only `current_hand_abs_13` plus the image into the policy. It then uses
the full `robot_state` to convert predicted relative actions back into absolute
commands:

```text
absolute_eef_target  = current_eef_abs composed with eef_relative
absolute_hand_target = current_hand_abs + hand_delta
```

The returned inference action is already absolute. Do not add it to another
base state again.

## Normalization

The diffusion-policy trainer computes normalization internally:

1. It builds absolute `[eef_6, hand_13]` state from the synced streams.
2. It samples horizon windows.
3. It converts each window to relative targets.
4. It fits a linear normalizer on those relative targets.
5. It trains diffusion in normalized space.
6. At inference, it unnormalizes predicted relative actions before converting
   them back to absolute commands.

Therefore:

- Save hand values in real command units.
- Save EEF values in real command units.
- Do not pre-normalize.
- Do not pre-scale for neural-network training.

## Recommended Raw/Processed Split

It is fine to keep richer raw recordings elsewhere, for example:

```text
<dataset_root>/raw/
  episode_000000/
    camera_original.mp4
    piper.csv
    proprioception.csv
    timestamps.json
    metadata.json
```

The diffusion trainer ignores that. The trainer only consumes:

```text
camera/mp4_files/*.mp4
processed_encoder_t/*.npy
absolute_action_t/*.npy
syncs/*.npz
```

Keep enough raw metadata to regenerate the processed arrays if the sync logic or
rotation convention changes.

## Validation Checklist

Before handing a dataset to diffusion training, validate every episode:

- Same number of `.mp4`, hand `.npy`, EEF `.npy`, and sync `.npz` files.
- Filenames sort into matching episode order.
- Video opens and has at least one frame.
- Hand array is rank 2 with shape `[T_hand, 13]`.
- EEF array is rank 2 with shape `[T_eef, 6]`.
- EEF orientation is rotation-vector, not unconverted roll/pitch/yaw.
- Sync file has `frame_idx`, `encoder_idx`, and `pose_idx`.
- Sync arrays are integer, 1D, same length, nonempty.
- Sync indices are all in range.
- No NaN or Inf values in hand or EEF arrays.
- Hand order and units are consistent across all episodes.
- EEF frame and units are consistent across all episodes.

Minimal validation snippet:

```python
from pathlib import Path
import cv2
import numpy as np

root = Path("data_trimorph")
videos = sorted((root / "camera" / "mp4_files").glob("*.mp4"))
hands = sorted((root / "processed_encoder_t").glob("*.npy"))
eefs = sorted((root / "absolute_action_t").glob("*.npy"))
syncs = sorted((root / "syncs").glob("*.npz"))

assert len(videos) == len(hands) == len(eefs) == len(syncs) > 0

for video_path, hand_path, eef_path, sync_path in zip(videos, hands, eefs, syncs):
    hand = np.load(hand_path)
    eef = np.load(eef_path)
    sync = np.load(sync_path)

    assert hand.ndim == 2 and hand.shape[1] == 13, (hand_path, hand.shape)
    assert eef.ndim == 2 and eef.shape[1] == 6, (eef_path, eef.shape)
    assert np.isfinite(hand).all(), hand_path
    assert np.isfinite(eef).all(), eef_path

    frame_idx = sync["frame_idx"].astype(np.int64)
    encoder_idx = sync["encoder_idx"].astype(np.int64)
    pose_idx = sync["pose_idx"].astype(np.int64)
    assert len(frame_idx) == len(encoder_idx) == len(pose_idx) > 0

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count > 0, video_path

    assert frame_idx.min() >= 0 and frame_idx.max() < frame_count
    assert encoder_idx.min() >= 0 and encoder_idx.max() < len(hand)
    assert pose_idx.min() >= 0 and pose_idx.max() < len(eef)
```

## Training Command Example

From the diffusion-policy repo:

```bash
python diffusion_policy/unet/train_unet_image_policy.py \
  --config-name=train_unet_image_trimorph_hand_cond_dinov2_policy \
  tasks.dataset.dataset_path=/path/to/data_trimorph
```

The task YAML owns the default dataset path:

```yaml
dataset_path: data_trimorph
```

Override it per collected dataset root, for example `data_trimorph1`,
`data_trimorph2`, or an absolute path.
