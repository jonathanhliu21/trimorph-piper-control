# Collection pipeline

`collect.py` launches exactly four workers: the OAK-1 policy camera, passive
PiperX logger, torque-disabled TRIMORPH leader, and commanded TRIMORPH
follower. All streams use host `monotonic_ns` timestamps.

The leader publishes its calibrated 13-motor `read_pos()` vector directly to
the follower. The follower clips and optionally smooths the target, commands
it with `set_positions()`, and records its own measured `read_pos()` values.
The 16-DOF `read_joint_pos()` expansion is intentionally not used.

There is no hand camera or MediaPipe path. The OAK-1 mounted at the robot wrist
is retained solely as the policy observation camera.
