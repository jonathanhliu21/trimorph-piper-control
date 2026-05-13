#!/usr/bin/env bash
set -e

CAN_IF="${1:-can0}"
BITRATE="${2:-1000000}"

sudo ip link set "$CAN_IF" down 2>/dev/null || true
sleep 0.2
sudo ip link set "$CAN_IF" txqueuelen 1000
sudo ip link set "$CAN_IF" type can bitrate "$BITRATE" restart-ms 100
sudo ip link set "$CAN_IF" up

ip -details -statistics link show "$CAN_IF"
