# Midas Piper Control

Small helper scripts for reading, recording, and replaying AgileX Piper arm
motion with `pyAgxArm`, plus Midas hand recording and replay examples.

## 1. Install Python Dependencies

Install `pyAgxArm` into your current Python environment. If you have the
`pyAgxArm` source checked out locally, run this from that checkout:

```bash
python3 -m pip install -e .
```

If CAN tools are not installed yet:

```bash
sudo apt update
sudo apt install can-utils ethtool
```

## 2. Start CAN

Plug in the Piper CAN adapter, then check that `can0` exists:

```bash
ip -details link show can0
```

If `can0` is not already up, start it at the Piper default bitrate:

```bash
sudo ip link set can0 up type can bitrate 1000000
```

For Piper recording, `examples/piper/setup_record.sh` brings up CAN and grants
the current user access to the PCsensor foot pedal input device:

```bash
examples/piper/setup_record.sh
```

To set up and immediately start the recorder:

```bash
examples/piper/setup_record.sh --start-record
```

The scripts default to:

- interface: `socketcan`
- channel: `can0`
- firmware driver: `default`

If your Piper firmware is newer, pass `--firmware v183` or `--firmware v188`.

## Piper API Wrapper

Main scripts can use the repo wrapper instead of calling the full `pyAgxArm`
API directly:

```python
from scripts.piper_api import PiperAPI

api = PiperAPI()
api.connect()
api.enable()

api.move_joint([0.0, 0.4, -0.4, 0.0, -0.4, 0.0])
api.move_joint([0.0, 0.4, -0.4, 0.0, -0.4, 0.0], mode="smooth")
api.move_eef([0.1, 0.0, 0.3, 0.0, 1.5708, 0.0])

api.disconnect()
```

By default, `PiperAPI()` sends the side-right installation profile when it
connects. `move_joint(...)` defaults to `mode="exact"`, which uses `move_js()`.
`mode="smooth"` uses `move_j()`. `move_eef(...)` commands the flange pose with
`move_p()`.

## Piper Gravity Compensation

The local gravity-compensation script uses MuJoCo to compute model-based torque
commands, then sends them through Piper MIT feed-forward torque control. Install
MuJoCo directly or install the Reimagine-Robotics gravity extras:

```bash
python3 -m pip install "piper_control[gravity]"
```

Bundled Piper arm models live under `assets/models/`:

- `piper_grav_comp.xml`: copied from Reimagine-Robotics/piper_control
- `piper_grav_comp_midas_hand.xml`: Piper arm + MIDAS hand kinematic payload (default)
- `midas_hand/midas_hand_kinematic.xml`: mesh-free hand model derived from
  `midas_hand_mujoco/assets/midas_description/midas_whole_hand.xml`
- `piper_grav_comp_payload_700g.xml`: simple 700 g box payload for quick tests

The default gravity-compensation model merges the Piper arm with the full MIDAS
hand inertial tree instead of a single `payload_tool` box. Tune the mount at
`gripper/midas_mount` in `piper_grav_comp_midas_hand.xml` if the physical
flange-to-palm transform differs.
Do not use `midas_hand_mujoco/assets/midas_description/midas_whole_hand.xml`
for arm gravity compensation; that file models the hand, not the Piper arm.

The CLI applies firmware-aware MIT scaling adapted from piper_control for
pyAgxArm:

- firmware `<= 1.8.post2`: send full MuJoCo torques for joints 1-3
- firmware `> 1.8.post2`: pre-multiply joints 1-3 by 4.0 before `move_mit`,
  because pyAgxArm always applies an internal 0.25 factor on those joints

Run a dry calculation first:

```bash
python3 gravity_compensate_piper.py --dry-run --zero-velocity-if-missing
```

Then test live output with a low torque scale:

```bash
python3 gravity_compensate_piper.py --torque-scale 0.2 --damping 1.0
```

Useful options:

```bash
python3 gravity_compensate_piper.py --help
python3 gravity_compensate_piper.py --model-path assets/models/piper_grav_comp.xml --dry-run
python3 gravity_compensate_piper.py --software-version S-V1.8-3 --iterations 100 --dry-run
```

## 3. Read Current State

From this repo:

```bash
python3 examples/piper/read_piper_state.py
```

This prints joint angles in radians and flange pose as:

```text
[x, y, z, roll, pitch, yaw]
```

Position is in meters. Orientation is in radians.

## 4. Record Motion

Record to the default file, `recorded_motion.csv`:

```bash
python3 examples/piper/record.py
```

The recorder disables all joint motors by default, then waits 3 seconds before
starting so you can get ready. It records:

- elapsed time
- robot timestamp
- joint angles `joint_1` through `joint_6`
- flange pose `x, y, z, roll, pitch, yaw`

Useful options:

```bash
python3 examples/piper/record.py --duration 10
python3 examples/piper/record.py --rate-hz 50
python3 examples/piper/record.py --output motions/wave.csv
python3 examples/piper/record.py --firmware v188 --channel can0
python3 examples/piper/record.py --keep-torque-on
```

Use `Ctrl+C` to stop an open-ended recording.

To use a USB foot pedal to start and stop recording, first find its Linux input
device:

```bash
python3 examples/piper/record.py --list-input-devices
```

Then pass the matching event device:

```bash
python3 examples/piper/record.py --foot-pedal-device /dev/input/event7
```

For the PCsensor pedal, prefer the stable symlink:

```bash
python3 examples/piper/record.py --foot-pedal-device /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
```

With the pedal enabled, the script still waits for teacher mode first. Press the
pedal once to start recording, then press it again to stop. If the pedal device
name contains `foot` or `pedal`, `--foot-pedal` can auto-detect it. If opening
the device fails with a permission error, run from a user that can read
`/dev/input/event*` devices.

## 5. Replay Motion

Replay the default file, `recorded_motion.csv`:

```bash
python3 examples/piper/replay.py
```

Replay a specific file:

```bash
python3 examples/piper/replay.py motions/wave.csv
```

Replay first moves slowly to the first recorded joint pose with `move_j()`, waits
briefly for the arm to settle, then plays the recorded timing with `move_js()`
by default. Use `--control-mode ik` to command the recorded flange pose with
`move_p()`, or use `--control-mode cpv` / `--control-mode move-j` to compare
against the other replay APIs. IK replay requires complete `x`, `y`, `z`,
`roll`, `pitch`, and `yaw` columns in the CSV.

Useful options:

```bash
python3 examples/piper/replay.py --speed-percent 50
python3 examples/piper/replay.py --preposition-speed-percent 15
python3 examples/piper/replay.py --preposition-seconds 8
python3 examples/piper/replay.py --settle-seconds 5
python3 examples/piper/replay.py --control-mode ik
python3 examples/piper/replay.py --control-mode cpv
python3 examples/piper/replay.py --firmware v188 --channel can0
python3 examples/piper/replay.py --no-enable
```

## Replay With Teacher-Mode Takeover

For DAgger-style collection, run:

```bash
python3 examples/piper/replay_with_takeover.py recorded_motion.csv
```

The script replays by time, not by pause/resume index. If takeover starts at
replay time 6s and lasts 1s, replay resumes at 7s in the original trajectory.

Default behavior:

- moves slowly to the first recorded pose before starting replay
- records one timestamped dataset under `scripts/recordings/`
- records replay rows and takeover rows in the same CSV
- enters takeover when the robot's teacher-mode button sets `teach_status` active
- waits 1 second after teacher mode starts, then records takeover rows
- exits takeover when teacher mode becomes inactive and resumes replay if the
  trajectory has not already ended
- can use an optional foot pedal as a stop button for replay or future inference

Useful options:

```bash
python3 examples/piper/replay_with_takeover.py --takeover-only
python3 examples/piper/replay_with_takeover.py --output scripts/recordings/test.csv
python3 examples/piper/replay_with_takeover.py --takeover-delay 0
python3 examples/piper/replay_with_takeover.py --speed-percent 20
python3 examples/piper/replay_with_takeover.py --foot-pedal-device /dev/input/event7
```

## Midas Hand Examples

The hand examples live under `examples/midas_hand/` and use the installed
`midas_hand_api` package. They load calibration from `~/.midas_hand/config.yaml`
by default, unless `--config` is passed.

Record motor positions, velocities, and currents:

```bash
python3 examples/midas_hand/record.py --output hand_recording.csv
```

Replay a recording:

```bash
python3 examples/midas_hand/replay.py hand_recording.csv --preposition
```

Move a single finger or thumb by editing the constants at the top of
`examples/midas_hand/move_finger.py`:

```python
SEQUENCE = [
    ["ring", 1.0, 0.0, 0.0],
    [
        ["index", 0.0, 0.0, 1.0],
        ["thumb", 0.5, 1.0, 2.0, 1.5],
    ],
]
WAIT_BETWEEN_MOVEMENTS_S = 0.5
```

Each command is `["finger", values...]`. Fingers use 3 values; the thumb uses
4 values. A nested list runs those commands in parallel before waiting. Then run:

```bash
python3 examples/midas_hand/move_finger.py
```

## Safety Notes

- Start with low `--speed-percent` values when testing replay.
- Keep the arm clear of people, cables, and hard stops.
- Make sure the selected `--firmware` matches the arm before sending motion commands.
- `examples/piper/record.py` sends a motor-disable command by default.
- Piper replay and Midas hand replay scripts send motion commands.
