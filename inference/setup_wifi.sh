#!/usr/bin/env bash
# Shared WiFi latency tuning for the realtime predictor<->executor link.
#
# WiFi power-save lets the radio doze between packets; waking it adds tens of ms
# of latency, which shows up as periodic spikes in ping (the enemy of a 15Hz
# control loop). This disables power-save immediately via `iw` (no reconnect)
# and writes a persistent NetworkManager drop-in so it survives reboots.
#
# Usage:
#   ./setup_wifi.sh                 # tune the autodetected interface
#   WIFI_IF=wlp8s0 ./setup_wifi.sh  # force a specific interface
#
# Or source it and call `tune_wifi` from another script.
set -euo pipefail

detect_wifi_if() {
    local ifc
    ifc="$(iw dev 2>/dev/null | awk '$1=="Interface"{print $2; exit}')"
    if [[ -z "${ifc}" ]]; then
        local d
        for d in /sys/class/net/*/wireless; do
            [[ -e "${d}" ]] || continue
            ifc="$(basename "$(dirname "${d}")")"
            break
        done
    fi
    printf '%s' "${ifc}"
}

tune_wifi() {
    local ifc="${WIFI_IF:-$(detect_wifi_if)}"
    if [[ -z "${ifc}" ]]; then
        echo "No wireless interface found. Set WIFI_IF=<iface> to override." >&2
        return 1
    fi

    echo "WiFi interface: ${ifc}"
    echo "  before: $(iw dev "${ifc}" get power_save 2>/dev/null || echo 'unknown')"

    # Immediate and non-disruptive: applies to the live link without a reconnect.
    sudo iw dev "${ifc}" set power_save off

    # Persistent: NetworkManager re-applies this on every future activation/reboot.
    if [[ -d /etc/NetworkManager ]]; then
        sudo tee /etc/NetworkManager/conf.d/wifi-powersave-off.conf >/dev/null <<'EOF'
[connection]
wifi.powersave = 2
EOF
        echo "  persistent: /etc/NetworkManager/conf.d/wifi-powersave-off.conf (wifi.powersave=2)"
    else
        echo "  NetworkManager not detected; skipped persistent config (iw setting holds until reboot)."
    fi

    echo "  after:  $(iw dev "${ifc}" get power_save 2>/dev/null || echo 'unknown')"
}

# Run directly when not sourced.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    sudo -v
    tune_wifi
fi
