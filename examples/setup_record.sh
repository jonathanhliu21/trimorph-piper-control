#!/usr/bin/env bash
set -euo pipefail

CAN_IF="${CAN_IF:-can0}"
BITRATE="${BITRATE:-1000000}"
PEDAL_DEVICE="${PEDAL_DEVICE:-/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd}"
START_RECORD="${START_RECORD:-0}"
TARGET_USER="${SUDO_USER:-${USER}}"

usage() {
    cat <<EOF
Usage: $0 [options] [-- record.py args...]

Options:
  --can-if NAME          CAN interface. Default: ${CAN_IF}
  --bitrate RATE         CAN bitrate. Default: ${BITRATE}
  --pedal-device PATH    Foot pedal input path. Default: ${PEDAL_DEVICE}
  --start-record         Start examples/piper/record.py after setup.
  -h, --help             Show this help.

Environment overrides:
  CAN_IF, BITRATE, PEDAL_DEVICE, START_RECORD

Examples:
  examples/piper/setup_record.sh
  examples/piper/setup_record.sh --start-record -- --output recorded_motion.csv
EOF
}

RECORD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --can-if)
            CAN_IF="$2"
            shift 2
            ;;
        --bitrate)
            BITRATE="$2"
            shift 2
            ;;
        --pedal-device)
            PEDAL_DEVICE="$2"
            shift 2
            ;;
        --start-record)
            START_RECORD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            RECORD_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

sudo -v

echo "Setting up ${CAN_IF} at ${BITRATE} bps..."
sudo ip link set "${CAN_IF}" down 2>/dev/null || true
sleep 0.2
sudo ip link set "${CAN_IF}" txqueuelen 1000
sudo ip link set "${CAN_IF}" type can bitrate "${BITRATE}" restart-ms 100
sudo ip link set "${CAN_IF}" up
ip -details -statistics link show "${CAN_IF}"

echo
echo "Checking foot pedal input..."
if [[ ! -e "${PEDAL_DEVICE}" ]]; then
    echo "Foot pedal device not found: ${PEDAL_DEVICE}" >&2
    echo "Available input devices:" >&2
    python3 "${REPO_ROOT}/examples/piper/record.py" --list-input-devices >&2
    exit 1
fi

PEDAL_REALPATH="$(readlink -f "${PEDAL_DEVICE}")"
echo "Foot pedal device: ${PEDAL_DEVICE} -> ${PEDAL_REALPATH}"

if ! sudo -u "${TARGET_USER}" test -r "${PEDAL_REALPATH}" || \
   ! sudo -u "${TARGET_USER}" test -w "${PEDAL_REALPATH}"; then
    echo "Granting ${TARGET_USER} access to ${PEDAL_REALPATH}..."
    if command -v setfacl >/dev/null 2>&1; then
        sudo setfacl -m "u:${TARGET_USER}:rw" "${PEDAL_REALPATH}"
    else
        echo "setfacl not found; using a temporary chmod fallback."
        sudo chmod a+rw "${PEDAL_REALPATH}"
    fi
fi

if ! sudo -u "${TARGET_USER}" test -r "${PEDAL_REALPATH}"; then
    echo "Still cannot read ${PEDAL_REALPATH} as ${TARGET_USER}. Try running this script again or check sudo output." >&2
    exit 1
fi

echo "Foot pedal access OK."
echo
echo "Record command:"
printf '  python3 %q --foot-pedal-device %q' \
    "${REPO_ROOT}/examples/piper/record.py" \
    "${PEDAL_DEVICE}"
if [[ ${#RECORD_ARGS[@]} -gt 0 ]]; then
    printf ' %q' "${RECORD_ARGS[@]}"
fi
printf '\n'

if [[ "${START_RECORD}" == "1" ]]; then
    echo
    echo "Starting recorder..."
    exec python3 "${REPO_ROOT}/examples/piper/record.py" \
        --foot-pedal-device "${PEDAL_DEVICE}" \
        "${RECORD_ARGS[@]}"
fi
