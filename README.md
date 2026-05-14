# Midas Piper Control

Small helper scripts for reading, recording, and replaying AgileX Piper arm motion with
`pyAgxArm`.

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

## 3. Read Current State

From this repo:

```bash
python3 examples/read_piper_state.py
```

This prints joint angles in radians and flange pose as:

```text
[x, y, z, roll, pitch, yaw]
```

Position is in meters. Orientation is in radians.

## 4. Record Motion

Record to the default file, `recorded_motion.csv`:

```bash
python3 examples/record.py
```

The recorder disables all joint motors by default, then waits 3 seconds before
starting so you can get ready. It records:

- elapsed time
- robot timestamp
- joint angles `joint_1` through `joint_6`
- flange pose `x, y, z, roll, pitch, yaw`

Useful options:

```bash
python3 examples/record.py --duration 10
python3 examples/record.py --rate-hz 50
python3 examples/record.py --output motions/wave.csv
python3 examples/record.py --firmware v188 --channel can0
python3 examples/record.py --keep-torque-on
```

Use `Ctrl+C` to stop an open-ended recording.

## 5. Replay Motion

Replay the default file, `recorded_motion.csv`:

```bash
python3 examples/replay.py
```

Replay a specific file:

```bash
python3 examples/replay.py motions/wave.csv
```

Replay first moves slowly to the first recorded joint pose with `move_j()`, waits
briefly for the arm to settle, then plays the recorded timing with `move_js()`
by default. Use `--control-mode ik` to command the recorded flange pose with
`move_p()`, or use `--control-mode cpv` / `--control-mode move-j` to compare
against the other replay APIs. IK replay requires complete `x`, `y`, `z`,
`roll`, `pitch`, and `yaw` columns in the CSV.

Useful options:

```bash
python3 examples/replay.py --speed-percent 50
python3 examples/replay.py --preposition-speed-percent 15
python3 examples/replay.py --preposition-seconds 8
python3 examples/replay.py --settle-seconds 5
python3 examples/replay.py --control-mode ik
python3 examples/replay.py --control-mode cpv
python3 examples/replay.py --firmware v188 --channel can0
python3 examples/replay.py --no-enable
```

## Replay With Keyboard Takeover

For DAgger-style collection, run:

```bash
python3 examples/replay_with_takeover.py recorded_motion.csv
```

The script replays by time, not by pause/resume index. If takeover starts at
replay time 6s and lasts 1s, replay resumes at 7s in the original trajectory.

Default behavior:

- moves slowly to the first recorded pose before starting replay
- records one timestamped dataset under `scripts/recordings/`
- records replay rows and takeover rows in the same CSV
- enters takeover when you press `Space` in the terminal
- disables all joint motors for takeover, waits 1 second, then records takeover rows
- exits takeover when you press `Space` again and resumes replay if the
  trajectory has not already ended

Useful options:

```bash
python3 examples/replay_with_takeover.py --takeover-only
python3 examples/replay_with_takeover.py --output scripts/recordings/test.csv
python3 examples/replay_with_takeover.py --takeover-delay 0
python3 examples/replay_with_takeover.py --speed-percent 20
python3 examples/replay_with_takeover.py --keyboard-toggle-debounce-seconds 0.5
```

## Safety Notes

- Start with low `--speed-percent` values when testing replay.
- Keep the arm clear of people, cables, and hard stops.
- Make sure the selected `--firmware` matches the arm before sending motion commands.
- `record.py` sends a motor-disable command by default. `replay.py` sends motion commands.
