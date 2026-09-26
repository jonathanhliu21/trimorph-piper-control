# Collection pipeline

`collect.py` always launches the passive PiperX logger and torque-disabled
TRIMORPH proprioception worker. The OAK-1 policy camera and a commanded MIDAS
follower are optional. All streams use host `monotonic_ns` timestamps.

The primary hand records its calibrated 13-motor `read_pos()` vector directly
to `proprioception.csv`. When configured, the follower clips and optionally
smooths that vector, commands it with `set_positions()`, and records its own
measured `read_pos()` values to `follower.csv`.
The 16-DOF `read_joint_pos()` expansion is intentionally not used.

There is no hand camera or MediaPipe path. The optional OAK-1 mounted at the
robot wrist is retained solely as the policy observation camera.
