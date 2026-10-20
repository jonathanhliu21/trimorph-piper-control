# Collection pipeline

`collect_hand_only.py` records the TRIMORPH leader hand. Add `--use-piper`
for passive PiperX logging, `--use-quest` for Quest UDP poses, `--add-camera`
for the OAK wrist camera, and both follower options for commanded MIDAS
retargeting. Piper and Quest pose sources are mutually exclusive.
All streams use host `monotonic_ns` timestamps. Foot-pedal recording is the
default; use `--episode-s` for a fixed duration or `--retarget-only` to run
without recording. `replay_episode.py` replays saved hand and Piper streams.

`collection/config.py` defines shared hardware and retargeting settings in
`HandRuntimeConfig` and collection-specific options in
`HandOnlyCollectionConfig`. Replay tuning lives in `collection/replay_config.py`.
Standalone calibration and diagnostics are under `scripts/`; dataset conversion
is `scripts/dataset/postprocess_dataset.py`. Run all commands from the repository
root. See [the project README](../README.md) for the full directory map.

The primary hand records its calibrated 13-motor `read_pos()` vector directly
to `leader_hand.csv`. When configured, the follower clips and optionally
smooths that vector, commands it with `set_positions()`, and records its own
measured `read_pos()` values to `follower.csv`.
The global `follower_command_scale` is multiplied by the motor-ID keyed
`follower_joint_command_scales`, which default to `0.6` for finger PIP joints, `0.5` for finger MCP joints,
and `1.0` for the other joints.
Collection and follower replay also add its motor-ID keyed
`follower_joint_command_offsets` after scaling. Each offset is the MIDAS
target angle when the corresponding TRIMORPH joint is at zero:
`MIDAS = TRIMORPH * global scale * joint scale + joint offset`.
The 16-DOF `read_joint_pos()` expansion is intentionally not used.
Replay and postprocessing still accept legacy episodes containing
`proprioception.csv`.

## Thumb retargeting calibration

Collect 40 paired thumb poses with both hands connected. MIDAS joints 0 and 1
remain at zero while joints 2 and 3 move through a joint-limit-aware
calibration set. The TRIMORPH hand remains torque-disabled. At each prompt,
hold the TRIMORPH thumb at the pose corresponding to the displayed MIDAS pose
and press Enter:

```bash
python scripts/thumb_retargeting/retarget_thumb.py \
  --trimorph-config path/to/trimorph.yaml \
  --trimorph-port /dev/serial/by-id/TRIMORPH \
  --midas-config config/midas_hand_config.yaml \
  --midas-port /dev/serial/by-id/MIDAS
```

This checkpoints `thumb_tip_matching.csv` after every point. Fit the affine
map from TRIMORPH joints 2--3 to MIDAS joints 2--3:

```bash
python scripts/thumb_retargeting/fit_model_to_endpoints.py \
  --input thumb_tip_matching.csv \
  --output thumb_retarget_model.json
```

`collect_hand_only.py` loads `thumb_retarget_model.json` by default whenever a
follower is configured. Use `--thumb-model PATH` to select another model.
Joints 0--1 use the normal action scaling; model predictions replace only
joints 2--3. All MIDAS commands are clipped to the configured limits before
the normal interpolation and maximum-step safety logic is applied.

There is no hand camera or MediaPipe path. The OAK-1 mounted at the
robot wrist is retained solely as the policy observation camera. It is enabled
for hand-only collection with `--add-camera`, which also opens a live preview
and records the standard `camera.mp4` and `camera_timestamps.csv` episode
artifacts. Press `q` or Escape in the preview to stop and save. Brightness and
50/60 Hz flicker prevention are controlled by the shared task-camera settings
in `collection/config.py` and are applied before the DepthAI pipeline starts.
Set `TASK_CAMERA_FLIP = "180"` there to rotate both the preview and recording.
