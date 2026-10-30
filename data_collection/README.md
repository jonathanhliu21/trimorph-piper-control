# Collection pipeline

`collect_hand_only.py` records the TRIMORPH leader hand. Add `--use-piper`
for passive PiperX logging, `--use-quest` for Quest UDP poses, `--add-camera`
for the OAK wrist camera, and both follower options for commanded MIDAS
retargeting. Piper and Quest pose sources are mutually exclusive.
All streams use host `monotonic_ns` timestamps. Foot-pedal recording is the
default; use `--episode-s` for a fixed duration or `--retarget-only` to run
without recording. `replay_episode.py` replays saved hand and Piper streams.

Collect only the primary hand and Quest for a 30-second episode (replace the serial-port placeholder):

```bash
python3 data_collection/collect_hand_only.py \
  --dataset-root data_trimorph_quest \
  --hand-config config/xl330_test_config.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --use-quest --quest-jump-threshold 0.05 \
  --episode-s 30
```

Omit `--episode-s 30` for the default foot-pedal episode control. Camera and Piper workers stay disabled unless their respective flags are supplied. Start the Quest sender streaming to the collector's configured UDP endpoint (default `0.0.0.0:5005`). With Quest enabled, the collector prints flushed `POSITION JUMP` warnings when consecutive received packets move farther than `--quest-jump-threshold` in native position units. Each warning includes a worker-lifetime count, time since the first packet, sequence number, step distance, packet interval, and XYZ delta. Checks include packets received between episodes, while CSV recording remains limited to active episodes. Samples are recorded unchanged. The threshold is saved in episode metadata; a flagged step is a review cue, not a tracking-loss diagnosis.

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

## Postprocess Quest episodes and replay

Convert Quest motion using a calibration from `scripts/calib/get_coord_transform.py`:

```bash
python3 data_collection/postprocess_dataset.py data_trimorph_quest \
  --episode 0 --calibration-json calibration.json --no-camera
```

Omit `--episode` to process every complete episode. `-f` / `--force` overwrites existing processed outputs. Omit `--calibration-json` to postprocess an existing raw Piper recording instead. Raw episode files, including `raw/episode_XXXXXX/piper.csv`, are preserved.

The output layout follows the Midas/Piper postprocessor:

- `processed_encoder_t/XXXXXX.npy`: float32 hand encoder array, shape `(N, 13)`.
- `absolute_action_t/XXXXXX.npy`: float32 Piper XYZ plus rotation vector in radians, shape `(M, 6)`.
- `camera/mp4_files/XXXXXX.mp4`: copied original video when camera processing is enabled.
- `syncs/XXXXXX.npz`: `frame_idx`, `encoder_idx`, and `pose_idx` mappings for camera-based synchronization. Camera timestamps are trimmed by one frame at each end by default; video frames retain their original numbering. Nearest matches outside `--max-sync-error-s` (default 0.010) are reported and retained, matching the reference processor.
- `processed/episode_XXXXXX/piper.csv`: full-rate replay poses with original Quest host timestamps and XYZ/roll/pitch/yaw radians, using the Piper API's Euler convention. Euler angles are unwrapped for continuity. This CSV's last three columns are **Euler angles**; the action array's last three columns are **rotation vectors**.
- `processed/episode_XXXXXX/metadata.json`: pose frame, source, calibration path/hash, anchors, and processing settings.
- `processing_summary.json`: per-episode output paths and counts.

`--no-camera` skips all camera reads, video copying, and OpenCV validation. It creates no video, and syncs use the hand timeline: `encoder_idx`, `pose_idx`, and `monotonic_ns`, with no fabricated `frame_idx`. Both camera and camera-free sync archives additionally retain full-stream `encoder_monotonic_ns` and `pose_monotonic_ns` arrays. Encoder/action arrays retain all samples; sync indices select the corresponding samples rather than resampling those arrays.

Calibration is relative: `p(t) = p_start + A @ (quest(t) - quest(0))`, with orientation changes mapped independently by `B @ Dq @ B.T`. No fitted absolute-origin offset is applied. A Quest-only recording contains no physical Piper starting pose, so the default output starts at zero position/identity rotation and is marked `relative_to_episode_start`. Although stored under the reference folder name `absolute_action_t`, these values use an episode-local origin; they are not absolute robot-base targets until anchored. Replay detects this metadata and anchors the full trajectory at the Piper's measured XYZ/RPY starting pose before prepositioning. Position deltas remain in Piper base axes; rotation changes are spatial rotations multiplied on the left. Missing/invalid measured starting feedback stops replay rather than commanding unanchored coordinates.

To generate absolute robot-base targets, supply a known physical starting pose:

```bash
python3 data_collection/postprocess_dataset.py data_trimorph_quest \
  --episode 0 --calibration-json calibration.json --no-camera \
  --piper-start-pose 0.4 0.0 0.3 0.0 0.0 0.0 -f
```

Those six example values are XYZ and roll/pitch/yaw radians; replace them with the intended Piper starting pose. Explicitly anchored output is marked `piper_base` and is replayed without runtime reanchoring.

Replay now reads `processed/episode_XXXXXX/piper.csv` by default, while the hand continues to use its original measured CSV:

```bash
python3 data_collection/replay_episode.py 0 --dataset-root data_trimorph_quest \
  --hand-config config/xl330_test_config.yaml --hand-port /dev/serial/by-id/TRIMORPH
```

The replay command moves the hardware. For the original raw Piper CSV behavior, add `--read-piper-raw`; this flag is required with `--joint-level`. Missing processed poses produce an error with no silent fallback to raw data. `--hand-only` does not require processed Piper files. The calibration quality and any Quest tracking discontinuities carry through into the converted data; postprocessing does not filter samples or establish workspace reachability.
