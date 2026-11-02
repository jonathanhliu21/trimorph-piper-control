# Collection pipeline

For the end-to-end workflow (preparing, collecting, checking, postprocessing,
and the output files) start with [PIPELINE.md](PIPELINE.md); this file holds
the details.

`collect_hand_only.py` records, by default, the TRIMORPH leader hand, Quest
UDP poses, ROS 2 Vicon mocap, and the OAK wrist camera. Turn the defaults off
with `--no-quest`, `--no-mocap`, or `--no-camera`. Optional streams: add
`--use-piper` for passive PiperX logging (it replaces the Quest as the pose
source; Piper and Quest are mutually exclusive), both follower options for
commanded MIDAS retargeting, and `--add-tactile` for Paxini sensing. With
`--retarget-only` nothing is recorded, so Quest and mocap default to off there.
All streams use host `monotonic_ns` timestamps; camera frames are stamped with
their DepthAI capture time on that clock. Foot-pedal recording is the
default; use `--episode-s` for a fixed duration or `--retarget-only` to run
without recording. `replay_episode.py` replays saved hand and Piper streams.

Each run saves its episodes to the next free `data_collection_n` folder
(`n = 0, 1, 2, ...`), printed at startup as `Saving episodes to ...`. Use
`--dataset-name NAME` to save to `NAME_n` instead; `NAME` may include parent
directories (`--dataset-name data/pick_cup` gives `data/pick_cup_0`, then
`data/pick_cup_1`, ...). Change the default name with `DEFAULT_DATASET_NAME` in
`collection/config.py`. To add episodes to an existing folder, pass
`--dataset-root PATH` instead. The folder is created when the first episode
starts. Pass the folder to `postprocess_dataset.py` and `replay_episode.py
--dataset-root`.

Collect only the primary hand and Quest (camera and mocap off) for a 30-second episode (replace the serial-port placeholder):

```bash
python3 data_collection/collect_hand_only.py \
  --hand-config config/xl330_test_config.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --no-camera --no-mocap --quest-jump-threshold 0.05 \
  --episode-s 30
```

Add `--piper-start-joints-deg J1 J2 J3 J4 J5 J6` (or `--piper-start-home` for
the teleop home in `config/teleop_home.json`) to enable Piper and move it in joint
space to that pose before collection starts (Enter confirms; `--piper-start-speed-deg-s`,
default 10). The arm keeps holding the pose while you collect, and its measured flange
pose is saved as `piper.start_pose` in every episode's `metadata.json`;
`postprocess_dataset.py` uses it as the start pose when `--piper-start-pose` is omitted.

Omit `--episode-s 30` for the default foot-pedal episode control. The camera, Quest, and mocap workers run unless disabled; the Piper worker runs only with `--use-piper`. Start the Quest sender streaming to the collector's configured UDP endpoint (default `0.0.0.0:5005`). With Quest enabled, the collector prints flushed `POSITION JUMP` warnings when consecutive received packets move farther than `--quest-jump-threshold` in native position units. Each warning includes a worker-lifetime count, time since the first packet, sequence number, step distance, packet interval, and XYZ delta. Checks include packets received between episodes, while CSV recording remains limited to active episodes. Samples are recorded unchanged. The threshold is saved in episode metadata; a flagged step is a review cue, not a tracking-loss diagnosis.

### Live tracking monitor

When Quest or mocap is recorded, a `trimorph_tracking` window shows both poses
in 3D, with one status box per stream and a REC indicator. A box is green
(`OK`, with the receive rate) while samples arrive and red otherwise:

| Red status | Meaning |
| --- | --- |
| `NO DATA` | nothing received yet (for Vicon: no pose topic yet) |
| `NO SIGNAL n s` | no sample for over 0.5 s (Vicon Wi-Fi gaps up to ~0.3 s are normal) |
| `OBJECT LOST` | Vicon is publishing an all-zero pose: the marker plate is not tracked |
| `POSITION FROZEN` | the Quest position moved less than 0.1 mm in 1 s: it lost positional tracking and keeps resending its last position (also shown while the controller lies still) |

Each stream is drawn relative to its pose when the window opened, with a 10 s
trail and an XYZ triad for the rotation since then. Without a calibration each
tracker uses its own axes; `--monitor-calibration calibration.json` maps the
Quest (`linear_matrix`) and, if present, Vicon (`vicon_reference`) motion to
Piper base axes. Drag to rotate, right-drag or scroll to zoom, double-click to
re-center; `q`/Escape stops collection as in the camera view. Disable it with
`--no-monitor`. The window only displays; it does not change what is recorded.

`collection/config.py` defines shared hardware and retargeting settings in
`HandRuntimeConfig` and collection-specific options in
`HandOnlyCollectionConfig`. Replay tuning lives in `collection/replay_config.py`.
Standalone calibration and diagnostics are under `scripts/`. Dataset conversion
is `data_collection/postprocess_dataset.py`, which extends the Piper-only
`scripts/dataset/postprocess_dataset.py`; [DATASET_SPEC.md](DATASET_SPEC.md)
defines the format the diffusion-policy trainer reads. Run all commands from the
repository root. See [the project README](../README.md) for the full directory map.

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
by default (disable with `--no-camera`) and opens a live preview
and records the standard `camera.mp4` and `camera_timestamps.csv` episode
artifacts. Press `q` or Escape in the preview to stop and save. Exposure, ISO,
white balance and frame rate are fixed (no auto-exposure or auto white balance)
by the shared task-camera settings in `collection/config.py` and are applied
before the DepthAI pipeline starts.
Set `TASK_CAMERA_FLIP = "180"` there to rotate both the preview and recording.

In `camera_timestamps.csv`, `monotonic_ns` is the DepthAI mid-exposure capture
time converted to the host clock, which synchronization uses.
`receipt_monotonic_ns` is when the frame reached the host; the difference is the
transfer and queueing latency (about 19 ms median, 30 ms p95, on the OAK-1).
`sequence_num` is the device frame counter, so a gap means a dropped frame.
Episodes recorded before this change stored the receipt time in `monotonic_ns`.
The default 1280x720 recording is 16:9, while the trainer resizes frames to
320x240 (4:3); see [DATASET_SPEC.md](DATASET_SPEC.md).

## Postprocess Quest episodes and replay

Convert Quest motion using a calibration from `scripts/calib/get_coord_transform.py`:

```bash
python3 data_collection/postprocess_dataset.py data_collection_0 \
  --episode 0 --calibration-json calibration.json --no-camera
```

Omit `--episode` to process every complete episode. `-f` / `--force` overwrites existing processed outputs. Omit `--calibration-json` to postprocess an existing raw Piper recording instead. Raw episode files, including `raw/episode_XXXXXX/piper.csv`, are preserved.

The output layout follows the Midas/Piper postprocessor:

- `processed_encoder_t/XXXXXX.npy`: float32 hand encoder array, shape `(N, 13)`.
- `absolute_action_t/XXXXXX.npy`: float32 Piper XYZ plus rotation vector in radians, shape `(M, 6)`.
- `camera/mp4_files/XXXXXX.mp4`: copied original video when camera processing is enabled.
- `syncs/XXXXXX.npz`: `frame_idx`, `encoder_idx`, and `pose_idx` mappings for camera-based synchronization, plus `start_pose` and `pose_frame` (below). Each camera frame is paired with the hand and pose samples nearest to it in time. Camera timestamps are trimmed by one frame at each end by default; video frames retain their original numbering. A pairing farther apart than `--max-sync-error-s` (default 0.010 s) is reported as a warning and kept; nothing is dropped or interpolated. The hand is sampled every 20 ms, so its nearest sample is normally 0–10 ms away, and the warning mostly flags timing jitter or gaps.
- `processed/episode_XXXXXX/piper.csv`: full-rate replay poses with original Quest host timestamps and XYZ/roll/pitch/yaw radians, using the Piper API's Euler convention. Euler angles are unwrapped for continuity. This CSV's last three columns are **Euler angles**; the action array's last three columns are **rotation vectors**.
- `processed/episode_XXXXXX/metadata.json`: pose frame, source, calibration path/hash, anchors, and processing settings.
- `processing_summary.json`: per-episode output paths and counts.

`--no-camera` skips all camera reads, video copying, and OpenCV validation. It creates no video, and syncs use the hand timeline: `encoder_idx`, `pose_idx`, and `monotonic_ns`, with no fabricated `frame_idx`. Both camera and camera-free sync archives additionally retain full-stream `encoder_monotonic_ns` and `pose_monotonic_ns` arrays. Encoder/action arrays retain all samples; sync indices select the corresponding samples rather than resampling those arrays.

Calibration is relative: `p(t) = p_start + A @ (quest(t) - quest(0))`, with orientation changes mapped independently by `B @ Dq @ B.T`. No fitted absolute-origin offset is applied. A Quest-only recording contains no physical Piper starting pose, so the default output starts at zero position/identity rotation and is marked `relative_to_episode_start`. Although stored under the reference folder name `absolute_action_t`, these values use an episode-local origin; they are not absolute robot-base targets until anchored. Replay detects this metadata and anchors the full trajectory at the Piper's measured XYZ/RPY starting pose before prepositioning. Position deltas remain in Piper base axes; rotation changes are spatial rotations multiplied on the left. Missing/invalid measured starting feedback stops replay rather than commanding unanchored coordinates.

Every sync archive records the episode's anchor: `pose_frame` is `piper_base` or `relative_to_episode_start`, and `start_pose` is the Piper-base start pose as `[x, y, z, rotvec]` (the action array's convention). With the default relative mapping, `absolute_action_t[0]` equals `start_pose`. For a raw Piper recording, `start_pose` is the first measured pose. A relative Quest episode has no physical start pose, so its `start_pose` is all NaN rather than a misleading zero. The same start pose is in `processed/episode_XXXXXX/metadata.json` as `piper_start_pose` (XYZ + roll/pitch/yaw).

To generate absolute robot-base targets, supply a known physical starting pose:

```bash
python3 data_collection/postprocess_dataset.py data_collection_0 \
  --episode 0 --calibration-json calibration.json --no-camera \
  --piper-start-pose 0.4 0.0 0.3 0.0 0.0 0.0 -f
```

Those six example values are XYZ and roll/pitch/yaw radians; replace them with the intended Piper starting pose. Explicitly anchored output is marked `piper_base` and is replayed without runtime reanchoring.

For a calibration fitted with `get_coord_transform.py --fit-offset`, `--piper-start-pose` is required. The postprocessor applies the additional rotating position correction `(R_piper(t) - R_piper_start) @ tracked_point_offset.vector`. Its value depends on the actual flange starting orientation; processing with an assumed identity orientation and later reanchoring would give incorrect positions. Calibration files without a tracked-point offset retain the relative replay behavior described above.

Replay now reads `processed/episode_XXXXXX/piper.csv` by default, while the hand continues to use its original measured CSV:

```bash
python3 data_collection/replay_episode.py 0 --dataset-root data_collection_0 \
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
python3 data_collection/collect_hand_only.py --episode-s 10
python3 data_collection/replay_episode.py 0 --hand-only \
  --use-mocap --mocap-output replay_mocap.csv
```

Add your usual hand/dataset arguments as needed. Collection writes
`raw/episode_XXXXXX/mocap.csv`, and records its path, row count, and topic names
in episode metadata. Mocap can be combined with Piper or Quest logging.
Replay records **live** mocap during the replay interval, excluding prepositioning
and settling, and saves partial data on Ctrl-C or an exception. It does not use
mocap to command the hardware. Retarget-only mode does not save episodes.

With mocap enabled, the recorder discovers all topics advertising the two
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

When the running Python cannot import Foxy's `rclpy` (the Python 3.12 `.venv`),
collection and replay start `scripts/calib/mocap_stream.py --raw-rows` under
`.venv-mocap` after sourcing `ros2/mocap_env.sh`, and receive the same rows from it.

To demonstrate with the Vicon marker plate of the calibration (the mount
carried by hand) instead of the Quest, collect with `--mocap-topic
/vicon/Yori/Yori` and postprocess with `--pose-source mocap --mocap-topic
/vicon/Yori/Yori` and a calibration fitted with `fit_quest_calibration.py
--reference mocap`. That calibration stores the Vicon-to-Piper transform and the
plate-to-flange offset, so every sample becomes the pose the flange would have
if the mount sat on the arm. Lost-object (all-zero) samples are dropped. Without
`--absolute` the first sample is pinned to the recorded Piper start pose; with it
the flange pose is replayed as measured, after checking that it starts within
`--max-start-offset-mm` and `--max-start-rotation-deg` of the start pose.

ROS timestamps may use a different clock from host receipt timestamps; no clock
offset is estimated. An enabled recording with no samples still writes a CSV
header. With `--no-mocap` (or `--retarget-only`), no ROS imports, discovery, initialization, or
subscriptions occur. Replay requires `--mocap-output` with `--use-mocap`;
`--mocap-topic` and `--mocap-output` require mocap to be enabled.
