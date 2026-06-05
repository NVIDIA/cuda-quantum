#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #
#
# Orchestrates the cpu_roce DeviceCallChannel tests over a two-port RoCE
# loopback.  Two modes:
#
#   default       Run the GoogleTest fixture (CpuRoceDispatchTest).  The fixture
#                 itself spawns the cpu_roce_test_daemon and drives the channel
#                 by hand-calling the device_call ABI.
#   --app         Run the compiler-driven proof: nvq++-compile a real __qpu__
#                 app (cpu_roce_device_call_app.cpp) whose cudaq::device_call is
#                 lowered to the realtime ABI, spawn the daemon, run the app with
#                 --cudaq-device-call=cpu_roce, and check it prints 42.  This is
#                 the cpu_roce analogue of the NVQPP device_call lit tests that
#                 prove the shared-memory / host-dispatch channels.
#
# In both modes this script optionally (1) configures the two NIC ports and
# (2) builds the needed targets.  Mirrors the spirit of hsb_test_cpu.sh.
#
# Usage:
#   cpu_roce_device_call_test.sh \
#       --channel-device mlx5_0 --channel-ip 10.0.0.1 \
#       --daemon-device  mlx5_1 --daemon-ip  10.0.0.2 \
#       [--app] [--setup-network] [--build] [--build-dir DIR] [--mtu 4200]
#
# A loopback cable between the two ports (or two ports on the same NIC) is
# assumed.  The two ports must be different so the caller and the service can
# each own a QP/GID.

set -euo pipefail

CHANNEL_DEVICE="${CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE:-}"
CHANNEL_IP="${CUDAQ_CPU_ROCE_TEST_CHANNEL_IP:-10.0.0.1}"
DAEMON_DEVICE="${CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE:-}"
DAEMON_IP="${CUDAQ_CPU_ROCE_TEST_DAEMON_IP:-10.0.0.2}"
BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)/build"
MTU=4200
DO_SETUP_NETWORK=0
DO_BUILD=0
DO_APP=0
GTEST_FILTER="CpuRoceDispatchTest.*"
# Ring geometry shared by the --app caller (channel) and the daemon (service).
# slot-size MUST equal the daemon's page-size and slots MUST be <= num-pages so
# the channel's per-slot RDMA writes land in valid daemon ring slots.
APP_SLOTS=64
APP_SLOT_SIZE=384

while [[ $# -gt 0 ]]; do
  case "$1" in
    --channel-device) CHANNEL_DEVICE="$2"; shift 2;;
    --channel-ip)     CHANNEL_IP="$2"; shift 2;;
    --daemon-device)  DAEMON_DEVICE="$2"; shift 2;;
    --daemon-ip)      DAEMON_IP="$2"; shift 2;;
    --build-dir)      BUILD_DIR="$2"; shift 2;;
    --mtu)            MTU="$2"; shift 2;;
    --filter)         GTEST_FILTER="$2"; shift 2;;
    --setup-network)  DO_SETUP_NETWORK=1; shift;;
    --build)          DO_BUILD=1; shift;;
    --app)            DO_APP=1; shift;;
    -h|--help)
      sed -n '2,46p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

ib_to_netdev() {
  ls "/sys/class/infiniband/$1/device/net/" 2>/dev/null | head -1 || true
}

# After `ip addr add`, the kernel creates the IPv4-mapped RoCEv2 GID a moment
# later (notably slower right after a cold boot), and the transceiver's GID
# lookup fails if it runs before the GID appears.  Poll the IB device's GID
# table until an IPv4-mapped ("...:ffff:....:....") RoCEv2 GID shows up, and
# re-assert the address if a transient event dropped it.  This is the step
# plain `ip addr add` was missing after a power cycle.
wait_for_roce_gid() {
  local ib_dev="$1" netdev="$2" ip="$3"
  # Match the IPv4-mapped GID for THIS exact address (e.g. 10.0.0.1 ->
  # ":ffff:0a00:0001"), not just any IPv4-mapped v2 GID, so we don't lock onto
  # the wrong SGID on a multi-IP port.
  local a b c d want
  IFS=. read -r a b c d <<<"$ip"
  want=$(printf ":ffff:%02x%02x:%02x%02x" "$a" "$b" "$c" "$d")
  local deadline=$((SECONDS + 20)) i g t
  while ((SECONDS < deadline)); do
    if ! ip -o -4 addr show dev "$netdev" 2>/dev/null | grep -q "${ip}/"; then
      sudo ip addr add "${ip}/24" dev "$netdev" 2>/dev/null || true
    fi
    for i in $(seq 0 31); do
      g="$(cat "/sys/class/infiniband/$ib_dev/ports/1/gids/$i" 2>/dev/null)" || continue
      t="$(cat "/sys/class/infiniband/$ib_dev/ports/1/gid_attrs/types/$i" 2>/dev/null)" || true
      if [[ "$g" == *"$want"* && "$t" == *"v2"* ]]; then
        echo "    $ib_dev: RoCEv2 IPv4 GID ready for $ip (gid[$i]=$g)"
        return 0
      fi
    done
    sleep 0.3
  done
  echo "    ERROR: $ib_dev: no RoCEv2 GID for $ip after 20s" >&2
  return 1
}

configure_port() {
  local ib_dev="$1" ip="$2"
  local netdev
  netdev="$(ib_to_netdev "$ib_dev")"
  if [[ -z "$netdev" ]]; then
    echo "ERROR: no netdev for IB device '$ib_dev'" >&2
    exit 1
  fi
  echo "  $ib_dev ($netdev): ip=$ip mtu=$MTU"
  # Drop the IP from any other interface to avoid duplicate-IP routing issues.
  for other in $(ls /sys/class/net/ 2>/dev/null); do
    if [[ "$other" != "$netdev" ]]; then
      sudo ip addr del "${ip}/24" dev "$other" 2>/dev/null || true
    fi
  done
  sudo ip link set "$netdev" up
  sudo ip link set "$netdev" mtu "$MTU" 2>/dev/null || true
  sudo ip addr flush dev "$netdev" 2>/dev/null || true
  sudo ip addr add "${ip}/24" dev "$netdev"
  # Block until the IPv4-mapped RoCEv2 GID is actually populated (see above),
  # otherwise the transceiver's GID lookup can race the kernel after a boot.
  wait_for_roce_gid "$ib_dev" "$netdev" "$ip"
}

if [[ -z "$CHANNEL_DEVICE" || -z "$DAEMON_DEVICE" ]]; then
  echo "ERROR: --channel-device and --daemon-device are required" >&2
  echo "       (available: $(ls /sys/class/infiniband/ 2>/dev/null | tr '\n' ' '))" >&2
  exit 1
fi

echo "=== cpu_roce device_call test ==="
echo "Channel: $CHANNEL_DEVICE @ $CHANNEL_IP"
echo "Daemon:  $DAEMON_DEVICE @ $DAEMON_IP"
echo "Build:   $BUILD_DIR"

if [[ "$DO_BUILD" == "1" ]]; then
  echo "--- Building cpu_roce_test_daemon + test_device_call_dispatch ---"
  cmake --build "$BUILD_DIR" --target cpu_roce_test_daemon test_device_call_dispatch
fi

if [[ "$DO_SETUP_NETWORK" == "1" ]]; then
  echo "--- Configuring network ---"
  configure_port "$CHANNEL_DEVICE" "$CHANNEL_IP"
  configure_port "$DAEMON_DEVICE" "$DAEMON_IP"
fi

# --app mode: nvq++-compile the device_call app, spawn the daemon ourselves
# (the app is not the GoogleTest fixture, so the script owns the daemon), run
# the app over cpu_roce, and verify it prints 42.  This exercises the actual
# compiler lowering (cudaq::device_call -> realtime acquire/dispatch/release),
# which the hand-written GoogleTest fixture bypasses.
run_app_test() {
  local script_dir nvqpp app_src daemon_bin libdir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  nvqpp="$BUILD_DIR/bin/nvq++"
  app_src="$script_dir/cpu_roce_device_call_app.cpp"
  daemon_bin="$BUILD_DIR/unittests/cpu_roce_test_daemon"
  libdir="$BUILD_DIR/lib"

  [[ -x "$nvqpp" ]] || { echo "ERROR: nvq++ not found: $nvqpp" >&2; return 1; }
  [[ -f "$app_src" ]] || { echo "ERROR: app source not found: $app_src" >&2; return 1; }
  [[ -x "$daemon_bin" ]] || { echo "ERROR: daemon not found: $daemon_bin (run with --build)" >&2; return 1; }

  local workdir; workdir="$(mktemp -d)"
  local daemon_pid=""
  # shellcheck disable=SC2317
  cleanup_app() {
    [[ -n "$daemon_pid" ]] && kill "$daemon_pid" 2>/dev/null || true
    rm -rf "$workdir" 2>/dev/null || true
  }
  trap cleanup_app RETURN

  echo "--- Compiling device_call app (nvq++ -frealtime-lowering) ---"
  "$nvqpp" --target qpp-cpu -frealtime-lowering --enable-mlir \
    "$app_src" -o "$workdir/cpu_roce_app" || {
    echo "ERROR: nvq++ compile failed" >&2; return 1; }

  echo "--- Starting daemon (service) on $DAEMON_DEVICE @ $DAEMON_IP ---"
  LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}" "$daemon_bin" \
    --device="$DAEMON_DEVICE" --local-ip="$DAEMON_IP" --rendezvous-port=0 \
    --num-pages="$APP_SLOTS" --page-size="$APP_SLOT_SIZE" --timeout=120 \
    > "$workdir/daemon.log" 2>&1 &
  daemon_pid=$!

  # Wait for the daemon to publish its rendezvous endpoint (or die).
  local port="" deadline=$((SECONDS + 20))
  while ((SECONDS < deadline)); do
    if ! kill -0 "$daemon_pid" 2>/dev/null; then
      echo "ERROR: daemon exited during startup:" >&2
      cat "$workdir/daemon.log" >&2; return 1
    fi
    port="$(sed -n 's/.*CPU_ROCE_DAEMON_READY port=\([0-9]*\).*/\1/p' \
              "$workdir/daemon.log" 2>/dev/null | head -1)"
    [[ -n "$port" ]] && break
    sleep 0.3
  done
  [[ -n "$port" ]] || { echo "ERROR: daemon never became ready" >&2; cat "$workdir/daemon.log" >&2; return 1; }
  echo "  daemon ready: rendezvous port=$port"

  echo "--- Running app over cpu_roce (channel $CHANNEL_DEVICE @ $CHANNEL_IP) ---"
  local out rc
  out="$(LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}" "$workdir/cpu_roce_app" \
    --cudaq-device-call=cpu_roce \
    --cudaq-device-call-slots="$APP_SLOTS" \
    --cudaq-device-call-slot-size="$APP_SLOT_SIZE" \
    "ib-device=$CHANNEL_DEVICE" "local-ip=$CHANNEL_IP" \
    "rendezvous-host=$DAEMON_IP" "rendezvous-port=$port" 2>&1)"
  rc=$?
  echo "$out"

  if [[ $rc -eq 0 ]] \
     && grep -q "device_call int result = 42" <<<"$out" \
     && grep -q "device_call measured integer result = 42" <<<"$out"; then
    echo "=== PASS: device_call over cpu_roce returned 42 (compiler-lowered) ==="
    return 0
  fi
  echo "=== FAIL: app rc=$rc or output missing the expected 42 results ===" >&2
  return 1
}

if [[ "$DO_APP" == "1" ]]; then
  run_app_test
  exit $?
fi

TEST_BIN="$BUILD_DIR/unittests/test_device_call_dispatch"
if [[ ! -x "$TEST_BIN" ]]; then
  echo "ERROR: test binary not found: $TEST_BIN (run with --build)" >&2
  exit 1
fi

export CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE="$CHANNEL_DEVICE"
export CUDAQ_CPU_ROCE_TEST_CHANNEL_IP="$CHANNEL_IP"
export CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE="$DAEMON_DEVICE"
export CUDAQ_CPU_ROCE_TEST_DAEMON_IP="$DAEMON_IP"

echo "--- Running $GTEST_FILTER ---"
exec "$TEST_BIN" --gtest_filter="$GTEST_FILTER" --gtest_color=yes
