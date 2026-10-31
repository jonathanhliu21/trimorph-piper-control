#!/usr/bin/env bash
set -eo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$workspace_dir")"
mocap_python="${MOCAP_PYTHON:-$repo_dir/.venv-mocap/bin/python}"
if [[ ! -x "$mocap_python" ]]; then
    mocap_python=/usr/bin/python3
fi
"$mocap_python" -c 'import sys; assert sys.version_info[:2] == (3, 8), "Foxy requires Python 3.8"'

source /opt/ros/foxy/setup.bash
cd "$workspace_dir"
"$mocap_python" /usr/bin/colcon build --symlink-install \
    --packages-select vicon_receiver trimorph_mocap \
    --cmake-args \
    -DPYTHON_EXECUTABLE="$mocap_python" \
    "$@"
