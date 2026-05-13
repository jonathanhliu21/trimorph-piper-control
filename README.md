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

Replay first moves slowly to the first recorded joint pose, allows 5 seconds for
that initial move, waits 3 seconds, then plays the recorded timing. It uses the
recorded joint angles and sends them with `move_j()`. The CSV also contains flange
pose columns, but those are saved for reference and are not directly commanded
during replay.

Useful options:

```bash
python3 examples/replay.py --speed-percent 50
python3 examples/replay.py --preposition-speed-percent 15
python3 examples/replay.py --preposition-seconds 8
python3 examples/replay.py --settle-seconds 5
python3 examples/replay.py --firmware v188 --channel can0
python3 examples/replay.py --no-enable
```

## Safety Notes

- Start with low `--speed-percent` values when testing replay.
- Keep the arm clear of people, cables, and hard stops.
- Make sure the selected `--firmware` matches the arm before sending motion commands.
- `record.py` sends a motor-disable command by default. `replay.py` sends motion commands.
