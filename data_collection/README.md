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

Add `--piper-start-joints-deg J1 J2 J3 J4 J5 J6` (or `--piper-start-home` for
the teleop home in `config/teleop_home.json`) to enable Piper and move it in joint
space to that pose before collection starts (Enter confirms; `--piper-start-speed-deg-s`,
default 10). The arm keeps holding the pose while you collect, and its measured flange
pose is saved as `piper.start_pose` in every episode's `metadata.json`;
`postprocess_dataset.py` uses it as the start pose when `--piper-start-pose` is omitted.

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

For a calibration fitted with `get_coord_transform.py --fit-offset`, `--piper-start-pose` is required. The postprocessor applies the additional rotating position correction `(R_piper(t) - R_piper_start) @ tracked_point_offset.vector`. Its value depends on the actual flange starting orientation; processing with an assumed identity orientation and later reanchoring would give incorrect positions. Calibration files without a tracked-point offset retain the relative replay behavior described above.

Replay now reads `processed/episode_XXXXXX/piper.csv` by default, while the hand continues to use its original measured CSV:

```bash
python3 data_collection/replay_episode.py 0 --dataset-root data_trimorph_quest \
  --hand-config config/xl330_test_config.yaml --hand-port /dev/serial/by-id/TRIMORPH
```

The replay command moves the hardware. For the original raw Piper CSV behavior, add `--read-piper-raw`; this flag is required with `--joint-level`. Missing processed poses produce an error with no silent fallback to raw data. `--hand-only` does not require processed Piper files. The calibration quality and any Quest tracking discontinuities carry through into the converted data; postprocessing does not filter samples or establish workspace reachability.

### Optional ROS 2 mocap CSV logging

The repository now includes a [ROS 2 workspace](../ros2/README.md). Build it with
`bash ros2/build.sh`, then source `/opt/ros/foxy/setup.bash` and
`ros2/install/local_setup.bash` from the repository root. The existing collection
and replay imports forward to the `trimorph_mocap` package.

Source the ROS 2 installation and the Vicon receiver workspace in the same shell
used to run collection/replay. The Python interpreter must be able to import
`rclpy` and the generated `vicon_receiver.msg` package. The `.msg` files under
`ros2/src/vicon_receiver/msg/` (also linked from `msg/`) describe the wire schema;
they do not themselves install Python messages
or identify the running publisher's topic names.

| Message | Fields and format | CSV behavior |
| --- | --- | --- |
| `vicon_receiver/msg/Position` | `std_msgs/Header`; `float32` translation `x_trans/y_trans/z_trans`; four `float32` rotation components `x_rot/y_rot/z_rot/w`; string subject, segment, translation type; `int32 frame_number` | One row per segment pose, preserving translation and quaternion components as received. |
| `vicon_receiver/msg/MarkerPositions` | `std_msgs/Header`; `uint32 frame_number`; array of `vicon_receiver/MarkerPosition` | One row per marker; array header and frame number repeated on every row. |
| `vicon_receiver/MarkerPosition` (nested) | Three `float32` translation values; string marker, subject, segment names; `bool occluded` | Position, names and occlusion recorded. Orientation columns are empty because markers have no orientation. Occluded markers are retained. |

The rotation fields are treated as quaternion components and are saved without
normalization or conversion. The definitions specify no translation units or
coordinate convention. Values remain in receiver units; `translation_type` is
preserved for segment poses. No orientation is computed for marker data.

```bash
python3 data_collection/collect_hand_only.py --use-mocap --episode-s 10
python3 data_collection/replay_episode.py 0 --hand-only \
  --use-mocap --mocap-output replay_mocap.csv
```

Add your usual hand/dataset arguments as needed. Collection writes
`raw/episode_XXXXXX/mocap.csv`, and records its path, row count, and topic names
in episode metadata. Mocap can be combined with Piper or Quest logging.
Replay records **live** mocap during the replay interval, excluding prepositioning
and settling, and saves partial data on Ctrl-C or an exception. It does not use
mocap to command the hardware. Retarget-only mode does not save episodes.

With `--use-mocap`, the recorder discovers all topics advertising the two
supported top-level ROS 2 types above, including topics that appear later.
To select particular topics, repeat `--mocap-topic /actual/topic/name` on either
command. An explicitly selected topic must advertise a supported type. Startup
fails after 10 seconds if no matching topic is discovered or a selected topic is
missing. Discovery does not guarantee that the publisher sends samples.

CSV columns include the source `topic`, `message_type`, `frame_id`, frame and
subject/segment/marker identifiers, all raw pose/marker fields, and:

- `ros_stamp_sec`, `ros_stamp_nanosec`: original ROS header timestamp, stored as
  separate integers to preserve precision. Nested markers share their array's stamp.
- `monotonic_ns`: host receipt time for synchronization with the hand/camera.
- `wall_time_s`: host Unix receipt time in seconds.
- `elapsed_s`: host receipt time relative to episode/replay start.
- `sample_idx`: sequential CSV row index; multiple markers in one message share
  timestamps but have separate row indices.

ROS timestamps may use a different clock from host receipt timestamps; no clock
offset is estimated. An enabled recording with no samples still writes a CSV
header. Without `--use-mocap`, no ROS imports, discovery, initialization, or
subscriptions occur. Replay requires `--mocap-output` with `--use-mocap`;
`--mocap-topic` and `--mocap-output` require mocap to be enabled.
