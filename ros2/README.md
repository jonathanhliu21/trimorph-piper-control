# ROS 2 mocap workspace

This workspace contains two packages:

- `vicon_receiver`: generates Python/C++ interfaces from `src/vicon_receiver/msg/`.
- `trimorph_mocap`: the subscriber, CSV recording helpers, and interactive viewer.

The repository's top-level `msg` link points to the canonical interface directory.
The original `scripts/calib/mocap_detection.py` and
`data_collection/collection/mocap_worker.py` paths forward to the ROS package,
so existing collection/replay commands and Python imports continue to work.

## Build with ROS 2 Foxy

From the repository root:

```bash
# First-time Python environment setup, if .venv-mocap does not exist:
/usr/bin/python3 -m venv --system-site-packages .venv-mocap
source .venv-mocap/bin/activate
python -m pip install 'matplotlib>=3.3'

bash ros2/build.sh
source ros2/mocap_env.sh
.venv-mocap/bin/python scripts/calib/mocap_detection.py --mocap-topic /vicon/Yori/Yori
```

`ros2/mocap_env.sh` sources Foxy and this workspace, and restricts Fast DDS to
the interface facing the Vicon network (default: the default-route interface;
pass a name such as `wlp0s20f3` to override). Without it, Fast DDS also
advertises Tailscale/Docker addresses, the remote publisher sends to those, and
delivery falls from ~200 Hz to near zero even though `ros2 topic list` still
shows the topic. Foxy's Fast DDS whitelists by IP only, so the script rewrites
`~/.ros/fastdds_mocap.xml` from the current address each time it is sourced;
re-source it after the IP changes.

The build script uses `.venv-mocap/bin/python` when available, otherwise system
Python. Both must be Python 3.8 for this machine's Foxy installation. Set
`MOCAP_PYTHON` to choose another Python 3.8 environment. The viewer requires
NumPy, SciPy, and Matplotlib 3.3 or newer; the installed system Matplotlib is too
old. Source the local workspace after any other Vicon workspace so these
interfaces take precedence.

The full collection/replay application still uses newer Python features (such
as `dataclass(kw_only=True)`, requiring Python 3.10+). This workspace packages
the standalone mocap components for Foxy's Python 3.8; it does not resolve that
existing interpreter conflict for the full application.

## Check remote messages

```bash
source ros2/mocap_env.sh
ros2 interface show vicon_receiver/msg/Position
ros2 topic info /vicon/Yori/Yori --verbose
timeout --signal=INT 15s ros2 topic echo /vicon/Yori/Yori \
  vicon_receiver/msg/Position --qos-reliability reliable
```

The timeout bounds a listener that would otherwise wait indefinitely. Exit code
124 means the deadline expired; inspect stdout to determine whether samples
arrived during that interval. Discovery and valid local type support alone do
not establish remote sample delivery. These interfaces must match the
publisher's actual definitions, and DDS traffic must pass between computers.
