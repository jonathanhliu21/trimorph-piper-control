#!/usr/bin/env bash
# Tune the WiFi link, then launch the action executor (the robot box).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Cache sudo creds up front so WiFi tuning doesn't prompt mid-run.
sudo -v

# shellcheck source=../setup_wifi.sh
source "${SCRIPT_DIR}/../setup_wifi.sh"
tune_wifi

echo
if [[ -f "${SCRIPT_DIR}/executor.py" ]]; then
    echo "Starting executor..."
    exec python3 -u "${SCRIPT_DIR}/executor.py" "$@"
else
    echo "WiFi tuned. (No executor.py yet — add it in ${SCRIPT_DIR} and re-run.)"
fi
