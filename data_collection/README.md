# Collection pipeline

`collect.py` always launches the passive PiperX logger and torque-disabled
TRIMORPH leader-hand worker and the OAK-1 policy camera. A commanded MIDAS
follower is optional. All streams use host `monotonic_ns` timestamps.

The primary hand records its calibrated 13-motor `read_pos()` vector directly
to `leader_hand.csv`. When configured, the follower clips and optionally
smooths that vector, commands it with `set_positions()`, and records its own
measured `read_pos()` values to `follower.csv`.
The global `follower_command_scale` is multiplied by the motor-ID keyed
`follower_joint_command_scales`, whose entries default to `1.0`.
Hand-only collection also adds its motor-ID keyed
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
python data_collection/retarget_thumb.py \
  --trimorph-config path/to/trimorph.yaml \
  --trimorph-port /dev/serial/by-id/TRIMORPH \
  --midas-config config/midas_hand_config.yaml \
  --midas-port /dev/serial/by-id/MIDAS
```

This checkpoints `thumb_tip_matching.csv` after every point. Fit the affine
map from TRIMORPH joints 2--3 to MIDAS joints 2--3:

```bash
python data_collection/fit_model_to_endpoints.py \
  --input thumb_tip_matching.csv \
  --output thumb_retarget_model.json
```

`collect_hand_only.py` loads `thumb_retarget_model.json` by default whenever a
follower is configured. Use `--thumb-model PATH` to select another model.
Joints 0--1 use the normal action scaling; model predictions replace only
joints 2--3. All MIDAS commands are clipped to the configured limits before
the normal interpolation and maximum-step safety logic is applied.

There is no hand camera or MediaPipe path. The OAK-1 mounted at the
robot wrist is retained solely as the policy observation camera.
