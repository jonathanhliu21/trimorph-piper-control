# Data collection and postprocessing pipeline

This guide follows one dataset from recording to the files the diffusion-policy
trainer reads. Run every command from the repository root with `.venv`
activated. Details live in [README.md](README.md) (collector options, replay,
mocap) and [DATASET_SPEC.md](DATASET_SPEC.md) (the trainer's contract).

```text
  TRIMORPH glove ─┐
  Quest (UDP) ────┤   collect_hand_only.py    raw/episode_N/     check_episodes.py    postprocess_dataset.py
  Vicon (ROS 2) ──┼──▶  one thread per   ──▶  CSV + MP4 per ──▶  quality report  ──▶  camera/  processed_encoder_t/
  OAK-1 camera ───┤     stream, host          stream, saved                           absolute_action_t/  syncs/
  Piper (opt.) ───┘     monotonic clock       on pedal release                        processed/  (replay)
```

## 1. Before collecting

| Check | How |
| --- | --- |
| Glove bus at 1 Mbaud, return delay 0 | Set on all 13 motors in Dynamixel Wizard; `baudrate: 1000000` in `config/xl330_test_config.yaml`. At 57600 baud the hand records 25 Hz instead of 50 Hz. |
| USB latency timer 1 ms | `cat /sys/bus/usb-serial/devices/ttyUSB*/latency_timer` prints 1 (udev rule `99-dynamixel-ft232h-latency.rules`). |
| Glove port | `ls /dev/serial/by-id/` shows the glove adapter (FTBEO9P0). |
| Quest streaming and tracking | The Quest sender targets UDP port 5005. The tracking monitor (below) must show the Quest box green while the controller moves. |
| Vicon plate visible | The monitor's VICON box is green, not `OBJECT LOST`. The collector starts the ROS 2 relay itself. |
| Start pose (needed for absolute or mocap poses) | Add `--piper-start-home`: the Piper moves to the teleop home after an Enter prompt, and its measured flange pose is saved in every episode. |

## 2. Collecting

```bash
python3 data_collection/collect_hand_only.py \
  --hand-config config/xl330_test_config.yaml \
  --hand-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO9P0-if00-port0
```

By default this records the leader hand, Quest, Vicon mocap, and the OAK-1 wrist
camera. Each run writes to a new folder `<name>_<n>`; the name defaults to
`DEFAULT_DATASET_NAME` in `collection/config.py` and is printed at startup
(`Saving episodes to ...`).

| Option | Effect |
| --- | --- |
| `--dataset-name NAME` | save this run to the next free `NAME_n` |
| `--dataset-root PATH` | append episodes to an existing folder |
| `--piper-start-home` | move the Piper to the start pose and record it |
| `--mocap-topic /vicon/Yori/Yori` | record only the hand-held marker plate |
| `--no-quest`, `--no-mocap`, `--no-camera` | drop a default stream |
| `--use-piper` | record Piper drag-teach poses instead of the Quest |
| `--episode-s 30` | one fixed-length episode instead of the pedal |
| `--monitor-calibration calibration.json` | show Quest (and Vicon, if calibrated) motion in Piper axes |

Episode control with the foot pedal:

| Action | Result |
| --- | --- |
| press | recording starts after 0.2 s |
| press and release | episode saved |
| hold 2 s | episode deleted |
| `q`/Escape in a window, or one Ctrl-C | collection stops; the current episode is saved |

Press Ctrl-C only once and wait for `Saved episode ...`. The streams are held in
memory and written only when an episode is saved; a second Ctrl-C during
shutdown, or closing the terminal, loses the episode except for its video.

Two windows show the run: the camera preview and the tracking monitor. The
monitor draws the Quest and Vicon paths in 3D with one status box per stream:
green with the receive rate while tracked, red with `NO SIGNAL`,
`OBJECT LOST` (Vicon cannot see the plate), or `POSITION FROZEN` (the Quest lost
positional tracking and repeats its last position). Do not demonstrate while a
box is red. Watch the terminal for repeated `read: ... no status packet` lines:
they mean the glove bus failed and the hand values are frozen copies.

## 3. What is recorded

Each saved episode is `raw/episode_XXXXXX/`. Every row carries `monotonic_ns`,
the host's monotonic clock shared by all streams, which is what synchronization
uses.

| File | Contents | Rate |
| --- | --- | --- |
| `leader_hand.csv` | TRIMORPH calibrated motor angles `pos_0..pos_12` (rad), passive, torque off | 50 Hz |
| `camera.mp4`, `camera_timestamps.csv` | OAK-1 1280x720 video; per frame `monotonic_ns` = DepthAI mid-exposure capture time, `receipt_monotonic_ns` = host arrival (~20 ms later), `sequence_num` | 30 Hz |
| `quest.csv` | raw controller packets `seq, px, py, pz, qx, qy, qz, qw`, unconverted | ~72 Hz |
| `mocap.csv` | Vicon `Position` rows (plate pose in mm, quaternion, `frame_number`) and `MarkerPositions` rows (each marker, `occluded`) | ~220 Hz received |
| `piper.csv` | Piper flange `x..yaw` (and joints with `--piper-joint-level`) | 250 Hz, only with `--use-piper` |
| `follower.csv` | MIDAS measured angles | only with a follower |
| `metadata.json` | status, stop reason, durations, sample counts, hardware settings and config hashes, `piper.start_pose`, errors | |

## 4. Checking

```bash
python3 scripts/dataset/check_episodes.py data_collection_1_1
```

The report gives each stream's rate, largest gap, and coverage, then flags:
hand rate below target, long runs of identical hand rows (still hand or bus
dropout), dropped camera frames, a frozen or jumping Quest, Vicon lost-object
samples, and episodes without `metadata.json` (stopped before saving). It ends
with a per-episode summary and exits 1 when anything is flagged. Delete episodes
that cannot be used before postprocessing; postprocessing stops at an episode
folder without `metadata.json`.

## 5. Postprocessing

```bash
# Quest as the arm pose source
python3 data_collection/postprocess_dataset.py data_collection_1_1 --calibration-json calibration.json
# Quest absolute in the Piper base, no start pose: a calibration with orientation.controller_rotation
# (fit_quest_calibration.py saves it) from the same Quest tracking session as the data
python3 data_collection/postprocess_dataset.py DATASET --calibration-json CAL --absolute

# Vicon marker plate as the source (needs a calibration fitted with --reference mocap);
# --absolute keeps the Vicon-observed flange pose in the Piper base and needs no start pose
python3 data_collection/postprocess_dataset.py DATASET --calibration-json CAL \
  --pose-source mocap --mocap-topic /vicon/Yori/Yori --absolute
```

Add `--episode N` for one episode and `-f` to overwrite. Raw files are never
modified. For each episode with `status: complete` the postprocessor:

1. **Loads the hand** from `leader_hand.csv` as a float32 `(N, 13)` array in
   motor order 0-12, unfiltered.
2. **Converts the arm pose to the Piper base frame.**
   - Quest: `p(t) = p_start + A (q(t) - q(0))` for position and
     `R(t) = B (Q(t) Q(0)^-1) B^T R_start` for orientation, with `A` and `B` from
     the calibration. Motion is relative to the first sample, so no Quest
     origin is needed.
   - Vicon: drops lost-object samples, re-times samples from Vicon frame
     numbers (Wi-Fi delivery is bursty), removes single-frame spikes and
     smooths (Gaussian, sigma 20 ms), then maps the plate pose to the flange
     with the calibration's Vicon-to-Piper and plate-to-flange transforms.
   - Piper (no calibration): the recorded flange pose as is.

   The start pose `p_start, R_start` is `--piper-start-pose`, else the pose
   recorded with `--piper-start-home`. Without either, a Quest episode starts at
   zero position and identity rotation and is marked
   `relative_to_episode_start`.
3. **Writes the action array** `absolute_action_t` as float32 `(M, 6)`:
   XYZ in metres plus a rotation vector (the trainer's SE(3) convention).
4. **Synchronizes.** One sync row per camera frame, dropping the first and last
   frame; each row takes the hand sample and the pose sample nearest in time to
   that frame. Pairs further apart than 10 ms (`--max-sync-error-s`) are
   reported but kept; nothing is interpolated.
5. **Validates** shapes, finiteness, index ranges, and video readability, then
   writes `processing_summary.json`.

## 6. Outputs

| Path | Contents | Used by |
| --- | --- | --- |
| `camera/mp4_files/XXXXXX.mp4` | copy of the episode video | trainer |
| `processed_encoder_t/XXXXXX.npy` | hand, float32 `(N, 13)`, rad | trainer |
| `absolute_action_t/XXXXXX.npy` | arm pose, float32 `(M, 6)`, XYZ m + rotation vector | trainer |
| `syncs/XXXXXX.npz` | `frame_idx`, `encoder_idx`, `pose_idx` (trainer); `encoder_monotonic_ns`, `pose_monotonic_ns`, `start_pose` (`[x, y, z, rotvec]`, NaN when unknown), `pose_frame` | trainer, analysis |
| `processed/episode_XXXXXX/piper.csv` | full-rate pose as XYZ + roll/pitch/yaw for replay | `replay_episode.py` |
| `processed/episode_XXXXXX/metadata.json` | pose source, frame, calibration path and hash, start pose | provenance |
| `processing_summary.json` | per-episode counts and paths | |

Train with `tasks.dataset.dataset_path=<dataset folder>` in the diffusion-policy
repository; replay with `replay_episode.py N --dataset-root <dataset folder>`.

## 7. Known limitations

- **Relative Quest poses.** Without a recorded start pose, `absolute_action_t`
  is motion relative to the episode's first sample (`pose_frame =
  relative_to_episode_start`, `start_pose` NaN). The trainer only uses relative
  windows, so training is unaffected; replay anchors the motion at the arm's
  current pose. Collect with `--piper-start-home` for Piper-base poses.
- **Calibration sessions.** A Quest calibration holds only for the Quest
  tracking session it was fitted in; recentering the Quest changes its frame.
  Refit when the Quest has been reset.
- **Mocap needs its own calibration.** `--pose-source mocap` requires
  `vicon_reference` in the calibration (`fit_quest_calibration.py RUN --reference
  mocap`), and a start pose unless `--absolute`.
- **Frozen hand reads are not marked.** A failed glove read silently repeats the
  last values for all 13 motors; only the terminal log and long identical runs
  in the check report reveal it.
- **Aspect ratio.** Video is 16:9 while the trainer resizes to 320x240 (4:3),
  squeezing the image horizontally.
- **Sync tolerance is a warning only.** With a 50 Hz hand and a 72 Hz Quest the
  nearest samples are within ~9 ms; the 10 ms warning mainly flags gaps.
