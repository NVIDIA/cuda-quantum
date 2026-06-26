#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Set up two local SoftRoCE (rdma_rxe) devices over a veth pair. This is a
# runner/host capability probe for CPU RoCE based two-process tests; it does not
# build or run CUDA-Q by itself. On success it prints the environment variables
# expected by unittests/device_call/cpu_roce_device_call_test.sh.

set -euo pipefail

NETDEV_A="${CUDAQ_SOFTROCE_NETDEV_A:-cudaq-rxe-a}"
NETDEV_B="${CUDAQ_SOFTROCE_NETDEV_B:-cudaq-rxe-b}"
RDEV_A="${CUDAQ_SOFTROCE_RDEV_A:-rxe_cudaq0}"
RDEV_B="${CUDAQ_SOFTROCE_RDEV_B:-rxe_cudaq1}"
NETNS_A="${CUDAQ_SOFTROCE_NETNS_A:-cudaq-rxe-ns-a}"
NETNS_B="${CUDAQ_SOFTROCE_NETNS_B:-cudaq-rxe-ns-b}"
IP_A="${CUDAQ_SOFTROCE_IP_A:-10.88.0.1}"
IP_B="${CUDAQ_SOFTROCE_IP_B:-10.88.0.2}"
PREFIX_LEN="${CUDAQ_SOFTROCE_PREFIX_LEN:-24}"
MTU="${CUDAQ_SOFTROCE_MTU:-4200}"
CLEANUP_ONLY=0
CLEANUP_FIRST=0
PRINT_EXPORTS=1
VERIFY_PINGPONG=0
USE_NETNS=0
SELF_LOOP=0
SKIP_RDMA_NETNS_MOVE=0
SOFTROCE_UNSUPPORTED_RC=77

usage() {
  cat <<EOF
Usage: setup_softroce_loopback.sh [options]

Options:
  --cleanup             Remove the SoftRoCE/veth devices and exit.
  --cleanup-first       Remove any previous devices before setup.
  --netdev-a NAME       First veth netdev (default: ${NETDEV_A}).
  --netdev-b NAME       Second veth netdev (default: ${NETDEV_B}).
  --rdev-a NAME         First RDMA device name (default: ${RDEV_A}).
  --rdev-b NAME         Second RDMA device name (default: ${RDEV_B}).
  --netns-a NAME       First network namespace (default: ${NETNS_A}).
  --netns-b NAME       Second network namespace (default: ${NETNS_B}).
  --ip-a ADDR           First IPv4 address (default: ${IP_A}).
  --ip-b ADDR           Second IPv4 address (default: ${IP_B}).
  --prefix-len N        IPv4 prefix length (default: ${PREFIX_LEN}).
  --mtu N               veth MTU (default: ${MTU}).
  --no-exports          Do not print cpu_roce test environment exports.
  --verify-pingpong     Run ibv_uc_pingpong over the RXE devices.
  --self-loop           Use one RXE device as both pingpong endpoints.
  --use-netns           Put each RXE endpoint in its own network namespace.
  --skip-rdma-netns-move
                        Diagnostic mode: create RXE from inside each netns,
                        but do not move the RDMA devices into those netns.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cleanup) CLEANUP_ONLY=1; shift ;;
    --cleanup-first) CLEANUP_FIRST=1; shift ;;
    --netdev-a) NETDEV_A="$2"; shift 2 ;;
    --netdev-b) NETDEV_B="$2"; shift 2 ;;
    --rdev-a) RDEV_A="$2"; shift 2 ;;
    --rdev-b) RDEV_B="$2"; shift 2 ;;
    --netns-a) NETNS_A="$2"; shift 2 ;;
    --netns-b) NETNS_B="$2"; shift 2 ;;
    --ip-a) IP_A="$2"; shift 2 ;;
    --ip-b) IP_B="$2"; shift 2 ;;
    --prefix-len) PREFIX_LEN="$2"; shift 2 ;;
    --mtu) MTU="$2"; shift 2 ;;
    --no-exports) PRINT_EXPORTS=0; shift ;;
    --verify-pingpong) VERIFY_PINGPONG=1; shift ;;
    --self-loop) SELF_LOOP=1; shift ;;
    --use-netns) USE_NETNS=1; shift ;;
    --skip-rdma-netns-move) SKIP_RDMA_NETNS_MOVE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ $(id -u) -eq 0 ]]; then
  SUDO=()
elif command -v sudo >/dev/null 2>&1; then
  SUDO=(sudo)
else
  echo "ERROR: this script needs root privileges or sudo." >&2
  exit 1
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command '$1' not found." >&2
    exit 1
  }
}

run() {
  echo "+ $*"
  "${SUDO[@]}" "$@"
}

ns_exists() {
  ip netns list 2>/dev/null | awk '{print $1}' | grep -qx "$1"
}

run_netns() {
  local ns="$1"
  shift
  echo "+ ip netns exec ${ns} $*"
  "${SUDO[@]}" ip netns exec "$ns" "$@"
}

load_rxe_module() {
  local kernel
  kernel="$(uname -r 2>/dev/null || true)"

  echo "=== SoftRoCE kernel module probe ==="
  echo "Kernel: ${kernel:-unknown}"

  if modinfo rdma_rxe >/dev/null 2>&1; then
    run modprobe rdma_rxe
    return 0
  fi
  if modinfo rxe >/dev/null 2>&1; then
    run modprobe rxe
    return 0
  fi

  echo "SOFTROCE_UNSUPPORTED: neither rdma_rxe nor rxe is available for this kernel." >&2
  echo "  kernel: ${kernel:-unknown}" >&2
  echo "  modules: /lib/modules/${kernel:-unknown}" >&2
  echo "SoftRoCE requires kernel RXE support; rdma-core/ibverbs-utils install userspace only." >&2
  echo "Install a matching kernel module package, build rdma_rxe for this VM kernel, or use a non-RDMA fallback transport in CI." >&2
  return "$SOFTROCE_UNSUPPORTED_RC"
}

configure_rdma_netns_mode() {
  echo "=== RDMA network namespace mode ==="
  rdma system show || true
  # Dedicated RDMA devices in network namespaces require exclusive mode before
  # the namespaces and RDMA devices are created.
  run rdma system set netns exclusive
  rdma system show || true
}

move_rdma_device_to_netns() {
  local dev="$1" ns="$2" log_file rc
  log_file="$(mktemp)"

  echo "+ rdma dev set ${dev} netns ${ns}"
  set +e
  "${SUDO[@]}" rdma dev set "$dev" netns "$ns" >"$log_file" 2>&1
  rc=$?
  set -e

  if [[ "$rc" == "0" ]]; then
    rm -f "$log_file"
    return 0
  fi

  cat "$log_file" >&2 || true
  if grep -qi "Operation not supported" "$log_file"; then
    cat >&2 <<EOF
SOFTROCE_UNSUPPORTED: rdma_rxe device '${dev}' cannot move into netns '${ns}'.
RXE netns isolation is unsupported on this kernel. This does not by itself mean
same-namespace SoftRoCE traffic is unsupported.
EOF
    rm -f "$log_file"
    return "$SOFTROCE_UNSUPPORTED_RC"
  fi

  rm -f "$log_file"
  return "$rc"
}

cleanup() {
  if command -v ip >/dev/null 2>&1; then
    if ns_exists "$NETNS_A"; then
      run_netns "$NETNS_A" rdma link delete "$RDEV_A" 2>/dev/null || true
    fi
    if ns_exists "$NETNS_B"; then
      run_netns "$NETNS_B" rdma link delete "$RDEV_B" 2>/dev/null || true
    fi
    run ip netns delete "$NETNS_A" 2>/dev/null || true
    run ip netns delete "$NETNS_B" 2>/dev/null || true
  fi
  if command -v rdma >/dev/null 2>&1; then
    run rdma link delete "$RDEV_A" 2>/dev/null || true
    run rdma link delete "$RDEV_B" 2>/dev/null || true
  fi
  if command -v ip >/dev/null 2>&1; then
    run ip link delete "$NETDEV_A" 2>/dev/null || true
    run ip link delete "$NETDEV_B" 2>/dev/null || true
  fi
}

ipv4_gid_suffix() {
  local a b c d
  IFS=. read -r a b c d <<<"$1"
  printf ':ffff:%02x%02x:%02x%02x' "$a" "$b" "$c" "$d"
}

read_maybe_netns() {
  local ns="$1" path="$2"
  if [[ -n "$ns" ]]; then
    "${SUDO[@]}" ip netns exec "$ns" cat "$path" 2>/dev/null || true
  else
    cat "$path" 2>/dev/null || true
  fi
}

roce_gid_index() {
  local rdev="$1" ip="$2" ns="${3:-}" want idx gid type
  want="$(ipv4_gid_suffix "$ip")"
  for idx in $(seq 0 31); do
    gid="$(read_maybe_netns "$ns" "/sys/class/infiniband/${rdev}/ports/1/gids/${idx}")"
    [[ -n "$gid" ]] || continue
    type="$(read_maybe_netns "$ns" "/sys/class/infiniband/${rdev}/ports/1/gid_attrs/types/${idx}")"
    if [[ "$gid" == *"$want"* && "$type" == *"v2"* ]]; then
      echo "$idx"
      return 0
    fi
  done
  return 1
}

wait_for_roce_gid() {
  local rdev="$1" ip="$2" ns="${3:-}" deadline gid_index gid
  deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    if gid_index="$(roce_gid_index "$rdev" "$ip" "$ns")"; then
      gid="$(read_maybe_netns "$ns" "/sys/class/infiniband/${rdev}/ports/1/gids/${gid_index}")"
      echo "${rdev}: RoCEv2 IPv4 GID ready for ${ip} (${gid_index}=${gid})"
      return 0
    fi
    sleep 0.3
  done
  echo "ERROR: ${rdev}: no RoCEv2 IPv4 GID for ${ip} after 20s" >&2
  return 1
}

show_network_diagnostics() {
  echo "=== SoftRoCE network diagnostics ==="
  if [[ "$USE_NETNS" == "1" ]]; then
    run_netns "$NETNS_A" ip route get "$IP_B" from "$IP_A" || true
    run_netns "$NETNS_B" ip route get "$IP_A" from "$IP_B" || true
    run_netns "$NETNS_A" ip route show table local || true
    run_netns "$NETNS_B" ip route show table local || true
  else
    ip route get "$IP_B" from "$IP_A" || true
    ip route show table local || true
  fi
}

verify_netns_ip_connectivity() {
  echo "=== Verifying netns IP connectivity with ping ==="
  if ! command -v ping >/dev/null 2>&1; then
    echo "WARNING: ping not found; skipping netns IP connectivity check." >&2
    return 0
  fi

  run_netns "$NETNS_A" ping -c 3 -W 1 "$IP_B"
  run_netns "$NETNS_B" ping -c 3 -W 1 "$IP_A"
}

show_infiniband_sysfs() {
  echo "=== /sys/class/infiniband visibility ==="
  if [[ "$USE_NETNS" == "1" ]]; then
    run_netns "$NETNS_A" sh -c 'ls -la /sys/class/infiniband || true'
    run_netns "$NETNS_B" sh -c 'ls -la /sys/class/infiniband || true'
  else
    ls -la /sys/class/infiniband || true
  fi
}

verify_pingpong() {
  require_cmd timeout
  require_cmd ibv_uc_pingpong

  local ns_a="" ns_b=""
  if [[ "$USE_NETNS" == "1" ]]; then
    ns_a="$NETNS_A"
    ns_b="$NETNS_B"
  fi

  local client_dev="$RDEV_A" client_ip="$IP_A"
  local server_dev="$RDEV_B" server_ip="$IP_B"
  if [[ "$SELF_LOOP" == "1" ]]; then
    server_dev="$RDEV_A"
    server_ip="$IP_A"
  fi

  local gid_a gid_b log_dir server_log client_log server_pid client_rc server_rc
  gid_a="$(roce_gid_index "$client_dev" "$client_ip" "$ns_a")"
  gid_b="$(roce_gid_index "$server_dev" "$server_ip" "$ns_b")"
  log_dir="$(mktemp -d)"
  server_log="${log_dir}/server.log"
  client_log="${log_dir}/client.log"

  echo "=== Verifying SoftRoCE UC traffic with ibv_uc_pingpong ==="
  echo "Server: ${server_dev} gid_index=${gid_b} ip=${server_ip}${ns_b:+ netns=${ns_b}}"
  echo "Client: ${client_dev} gid_index=${gid_a} ip=${client_ip}${ns_a:+ netns=${ns_a}}"

  if [[ "$USE_NETNS" == "1" ]]; then
    "${SUDO[@]}" ip netns exec "$NETNS_B" timeout 25s ibv_uc_pingpong \
      -d "$server_dev" -i 1 -g "$gid_b" -n 16 -s 64 >"$server_log" 2>&1 &
  else
    timeout 25s ibv_uc_pingpong -d "$server_dev" -i 1 -g "$gid_b" -n 16 -s 64 \
      >"$server_log" 2>&1 &
  fi
  server_pid=$!
  sleep 1
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "ERROR: ibv_uc_pingpong server exited before the client started" >&2
    cat "$server_log" >&2 || true
    rm -rf "$log_dir"
    return 1
  fi

  set +e
  if [[ "$USE_NETNS" == "1" ]]; then
    "${SUDO[@]}" ip netns exec "$NETNS_A" timeout 20s ibv_uc_pingpong \
      -d "$client_dev" -i 1 -g "$gid_a" -n 16 -s 64 "$server_ip" \
      >"$client_log" 2>&1
  else
    timeout 20s ibv_uc_pingpong -d "$client_dev" -i 1 -g "$gid_a" -n 16 -s 64 "$server_ip" \
      >"$client_log" 2>&1
  fi
  client_rc=$?
  wait "$server_pid"
  server_rc=$?
  set -e

  echo "--- ibv_uc_pingpong client log ---"
  cat "$client_log" || true
  echo "--- ibv_uc_pingpong server log ---"
  cat "$server_log" || true
  rm -rf "$log_dir"

  if [[ "$client_rc" != "0" || "$server_rc" != "0" ]]; then
    echo "ERROR: ibv_uc_pingpong failed (client=${client_rc}, server=${server_rc})" >&2
    return 1
  fi
}

setup_netns() {
  if ns_exists "$NETNS_A" || ns_exists "$NETNS_B"; then
    echo "ERROR: ${NETNS_A}/${NETNS_B} already exist. Use --cleanup-first." >&2
    exit 1
  fi

  run ip netns add "$NETNS_A"
  run ip netns add "$NETNS_B"
  run ip link add "$NETDEV_A" type veth peer name "$NETDEV_B"
  run ip link set "$NETDEV_A" netns "$NETNS_A"
  run ip link set "$NETDEV_B" netns "$NETNS_B"

  run_netns "$NETNS_A" ip link set lo up
  run_netns "$NETNS_B" ip link set lo up
  run_netns "$NETNS_A" ip link set "$NETDEV_A" mtu "$MTU"
  run_netns "$NETNS_B" ip link set "$NETDEV_B" mtu "$MTU"
  run_netns "$NETNS_A" ip addr add "${IP_A}/${PREFIX_LEN}" dev "$NETDEV_A"
  run_netns "$NETNS_B" ip addr add "${IP_B}/${PREFIX_LEN}" dev "$NETDEV_B"
  run_netns "$NETNS_A" ip link set "$NETDEV_A" up
  run_netns "$NETNS_B" ip link set "$NETDEV_B" up

  show_network_diagnostics
  verify_netns_ip_connectivity

  run_netns "$NETNS_A" rdma link add "$RDEV_A" type rxe netdev "$NETDEV_A"
  run_netns "$NETNS_B" rdma link add "$RDEV_B" type rxe netdev "$NETDEV_B"

  echo "=== RDMA visibility after netns-local RXE creation ==="
  rdma dev show || true
  run_netns "$NETNS_A" rdma dev show || true
  run_netns "$NETNS_B" rdma dev show || true
  run_netns "$NETNS_A" ibv_devices || true
  run_netns "$NETNS_B" ibv_devices || true
  show_infiniband_sysfs

  if [[ "$SKIP_RDMA_NETNS_MOVE" == "1" ]]; then
    echo "Skipping explicit rdma dev set ... netns move for diagnostic mode."
  else
    # RXE links can be created from a network namespace while the RDMA devices
    # remain visible in the initial RDMA namespace. Move them explicitly so
    # libibverbs opens the endpoint-local device in each process namespace.
    move_rdma_device_to_netns "$RDEV_A" "$NETNS_A"
    move_rdma_device_to_netns "$RDEV_B" "$NETNS_B"
  fi

  wait_for_roce_gid "$RDEV_A" "$IP_A" "$NETNS_A"
  wait_for_roce_gid "$RDEV_B" "$IP_B" "$NETNS_B"
}
if [[ "$CLEANUP_ONLY" == "1" ]]; then
  cleanup
  exit 0
fi

require_cmd ip
require_cmd rdma
require_cmd modprobe

if [[ "$CLEANUP_FIRST" == "1" ]]; then
  cleanup
fi

if [[ "$SELF_LOOP" == "1" && "$USE_NETNS" == "1" ]]; then
  echo "ERROR: --self-loop cannot be combined with --use-netns." >&2
  exit 1
fi

load_rxe_module

if [[ "$USE_NETNS" == "1" ]]; then
  configure_rdma_netns_mode
  setup_netns
elif [[ "$SELF_LOOP" == "1" ]]; then
  if ip link show "$NETDEV_A" >/dev/null 2>&1 || ip link show "$NETDEV_B" >/dev/null 2>&1; then
    echo "ERROR: ${NETDEV_A}/${NETDEV_B} already exist. Use --cleanup-first." >&2
    exit 1
  fi

  run ip link add "$NETDEV_A" type veth peer name "$NETDEV_B"
  run ip link set "$NETDEV_A" mtu "$MTU"
  run ip link set "$NETDEV_B" mtu "$MTU"
  run ip addr add "${IP_A}/${PREFIX_LEN}" dev "$NETDEV_A"
  run ip link set "$NETDEV_A" up
  run ip link set "$NETDEV_B" up

  run rdma link add "$RDEV_A" type rxe netdev "$NETDEV_A"

  wait_for_roce_gid "$RDEV_A" "$IP_A"
else
  if ip link show "$NETDEV_A" >/dev/null 2>&1 || ip link show "$NETDEV_B" >/dev/null 2>&1; then
    echo "ERROR: ${NETDEV_A}/${NETDEV_B} already exist. Use --cleanup-first." >&2
    exit 1
  fi

  run ip link add "$NETDEV_A" type veth peer name "$NETDEV_B"
  run ip link set "$NETDEV_A" mtu "$MTU"
  run ip link set "$NETDEV_B" mtu "$MTU"
  run ip addr add "${IP_A}/${PREFIX_LEN}" dev "$NETDEV_A"
  run ip addr add "${IP_B}/${PREFIX_LEN}" dev "$NETDEV_B"
  run ip link set "$NETDEV_A" up
  run ip link set "$NETDEV_B" up

  run rdma link add "$RDEV_A" type rxe netdev "$NETDEV_A"
  run rdma link add "$RDEV_B" type rxe netdev "$NETDEV_B"

  wait_for_roce_gid "$RDEV_A" "$IP_A"
  wait_for_roce_gid "$RDEV_B" "$IP_B"
fi

echo "=== SoftRoCE loopback devices ==="
show_network_diagnostics
if [[ "$USE_NETNS" == "1" ]]; then
  rdma dev show || true
  run_netns "$NETNS_A" rdma dev show || true
  run_netns "$NETNS_B" rdma dev show || true
  run_netns "$NETNS_A" rdma link show || true
  run_netns "$NETNS_B" rdma link show || true
  run_netns "$NETNS_A" ibv_devices || true
  run_netns "$NETNS_B" ibv_devices || true
  run_netns "$NETNS_A" ibv_devinfo -d "$RDEV_A" || true
  run_netns "$NETNS_B" ibv_devinfo -d "$RDEV_B" || true
else
  rdma link show || true
  if command -v ibv_devices >/dev/null 2>&1; then
    ibv_devices || true
  fi
  if command -v ibv_devinfo >/dev/null 2>&1; then
    ibv_devinfo -d "$RDEV_A" || true
    if [[ "$SELF_LOOP" != "1" ]]; then
      ibv_devinfo -d "$RDEV_B" || true
    fi
  fi
fi

if [[ "$VERIFY_PINGPONG" == "1" ]]; then
  verify_pingpong
fi

if [[ "$PRINT_EXPORTS" == "1" ]]; then
  if [[ "$SELF_LOOP" == "1" ]]; then
    RDEV_B="$RDEV_A"
    IP_B="$IP_A"
  fi
  cat <<EOF

# Environment for unittests/device_call/cpu_roce_device_call_test.sh:
# If --use-netns was used, run the channel-side process in ${NETNS_A}
# and the daemon-side process in ${NETNS_B}.
export CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE=${RDEV_A}
export CUDAQ_CPU_ROCE_TEST_CHANNEL_IP=${IP_A}
export CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE=${RDEV_B}
export CUDAQ_CPU_ROCE_TEST_DAEMON_IP=${IP_B}
EOF
fi
