# Source (do not execute) to set up ROS 2 Foxy + this workspace for Vicon mocap:
#
#     source ros2/mocap_env.sh            # auto-detect the default-route interface
#     source ros2/mocap_env.sh wlp0s20f3  # or name the interface facing the Vicon network
#
# Fast DDS otherwise advertises every interface (Tailscale, Docker, ...); the remote
# Vicon publisher then sends to unreachable addresses and delivery drops to ~0 Hz.
# Fast DDS 2.1 (Foxy) whitelists by IP only, so the profile is regenerated from the
# interface's current address each time this file is sourced.

if [ -n "${BASH_SOURCE[0]}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "mocap_env.sh must be sourced: source ${0}" >&2
    exit 1
fi

_mocap_ws="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_mocap_iface="${1:-$(ip -4 route show default | awk '{print $5; exit}')}"
_mocap_ip="$(ip -4 -o addr show dev "$_mocap_iface" 2>/dev/null | awk '{split($4, a, "/"); print a[1]; exit}')"

if [ -z "$_mocap_ip" ]; then
    echo "mocap_env.sh: no IPv4 address on interface '${_mocap_iface}'" >&2
else
    source /opt/ros/foxy/setup.bash
    source "$_mocap_ws/install/local_setup.bash"
    mkdir -p "$HOME/.ros"
    _mocap_profile="$HOME/.ros/fastdds_mocap.xml"
    cat > "$_mocap_profile" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>mocap_udp</transport_id>
      <type>UDPv4</type>
      <interfaceWhiteList><address>${_mocap_ip}</address></interfaceWhiteList>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="mocap" is_default_profile="true">
    <rtps>
      <userTransports><transport_id>mocap_udp</transport_id></userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
    </rtps>
  </participant>
</profiles>
EOF
    export FASTRTPS_DEFAULT_PROFILES_FILE="$_mocap_profile"
    export RMW_FASTRTPS_USE_QOS_FROM_XML=0
    # The CLI daemon caches discovery from before the whitelist; restart it.
    ros2 daemon stop >/dev/null 2>&1
    echo "mocap_env: Fast DDS restricted to ${_mocap_iface} (${_mocap_ip})"
fi
unset _mocap_ws _mocap_iface _mocap_ip _mocap_profile
