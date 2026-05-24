#!/usr/bin/env bash
# Tune the WiFi link, then launch the inference predictor (the GPU/5090 box).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Cache sudo creds up front so WiFi tuning doesn't prompt mid-run.
sudo -v

# shellcheck source=../setup_wifi.sh
source "${SCRIPT_DIR}/../setup_wifi.sh"
tune_wifi

echo
if [[ -f "${SCRIPT_DIR}/predictor.py" ]]; then
    echo "Starting predictor..."
    exec python3 "${SCRIPT_DIR}/predictor.py" "$@"
else
    echo "WiFi tuned. (No predictor.py yet — add it in ${SCRIPT_DIR} and re-run.)"
fi
