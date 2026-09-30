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
The 16-DOF `read_joint_pos()` expansion is intentionally not used.
Replay and postprocessing still accept legacy episodes containing
`proprioception.csv`.

There is no hand camera or MediaPipe path. The OAK-1 mounted at the
robot wrist is retained solely as the policy observation camera.
