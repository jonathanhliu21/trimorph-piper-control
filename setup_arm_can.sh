#!/usr/bin/env bash
set -euo pipefail

CAN_IF="${CAN_IF:-can0}"
BITRATE="${BITRATE:-1000000}"

usage() {
    cat <<EOF
Usage: $0 [options]

Configure and activate the SocketCAN interface used by the Piper arm.

Options:
  --can-if NAME     CAN interface. Default: ${CAN_IF}
  --bitrate RATE    CAN bitrate. Default: ${BITRATE}
  -h, --help        Show this help.

Environment overrides:
  CAN_IF, BITRATE

Examples:
  ./setup_arm_can.sh
  ./setup_arm_can.sh --can-if can0 --bitrate 1000000
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --can-if)
            if [[ $# -lt 2 ]]; then
                echo "--can-if requires a value" >&2
                exit 2
            fi
            CAN_IF="$2"
            shift 2
            ;;
        --bitrate)
            if [[ $# -lt 2 ]]; then
                echo "--bitrate requires a value" >&2
                exit 2
            fi
            BITRATE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "${BITRATE}" =~ ^[0-9]+$ ]] || [[ "${BITRATE}" == "0" ]]; then
    echo "Invalid CAN bitrate: ${BITRATE}" >&2
    exit 2
fi

if ! command -v ip >/dev/null 2>&1; then
    echo "The 'ip' command is required. Install the iproute2 package." >&2
    exit 1
fi

if ! ip link show "${CAN_IF}" >/dev/null 2>&1; then
    echo "CAN interface '${CAN_IF}' does not exist." >&2
    echo "Detected CAN interfaces:" >&2
    ip -brief link show type can >&2 || true
    exit 1
fi

sudo -v

echo "Configuring ${CAN_IF} at ${BITRATE} bps..."
sudo ip link set "${CAN_IF}" down 2>/dev/null || true
sudo ip link set "${CAN_IF}" txqueuelen 1000
sudo ip link set "${CAN_IF}" type can bitrate "${BITRATE}" restart-ms 100
sudo ip link set "${CAN_IF}" up

echo
echo "Piper CAN interface is ready:"
ip -details -statistics link show "${CAN_IF}"
