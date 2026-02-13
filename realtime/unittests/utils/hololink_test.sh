#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.   #
# ============================================================================ #
#
# hololink_test.sh
#
# Orchestration script for end-to-end Hololink RPC dispatch testing.
# Tests libcudaq-realtime dispatch kernel over Hololink RDMA with a
# simple increment RPC handler (no QEC or decoder dependency).
#
# Modes:
#   Default (FPGA):   bridge + playback  (requires real FPGA)
#   --emulate:        emulator + bridge + playback  (no FPGA needed)
#
# Actions (can be combined):
#   --build            Build all required tools
#   --setup-network    Configure ConnectX interfaces
#   (run is implicit unless only --build / --setup-network are given)
#
# Examples:
#   # Full emulated test: build, configure network, run
#   ./hololink_test.sh --emulate --build --setup-network
#
#   # Just run with real FPGA (tools already built, network already set up)
#   ./hololink_test.sh --fpga-ip 192.168.0.2
#
#   # Build only
#   ./hololink_test.sh --build --no-run
#
set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================

EMULATE=false
DO_BUILD=false
DO_SETUP_NETWORK=false
DO_RUN=true
VERIFY=true

# Directory defaults
HOLOLINK_DIR="/workspaces/cuda-qx/hololink"
CUDA_QUANTUM_DIR="/workspaces/cuda-quantum"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Network defaults
IB_DEVICE=""           # auto-detect
BRIDGE_IP="10.0.0.1"
EMULATOR_IP="10.0.0.2"
FPGA_IP="192.168.0.2"
MTU=4096

# Run defaults
GPU_ID=0
TIMEOUT=60
NUM_SHOTS=100
PAYLOAD_SIZE=8
PAGE_SIZE=384
NUM_PAGES=64
CONTROL_PORT=8193

# Build parallelism
JOBS=$(nproc 2>/dev/null || echo 8)

# ============================================================================
# Argument Parsing
# ============================================================================

print_usage() {
    cat <<'EOF'
Usage: hololink_test.sh [options]

Modes:
  --emulate              Use FPGA emulator (3-tool mode, no FPGA needed)
                         Default: FPGA mode (2-tool, requires real FPGA)

Actions:
  --build                Build all required tools before running
  --setup-network        Configure ConnectX network interfaces
  --no-run               Skip running the test (useful with --build)

Build options:
  --hololink-dir DIR     Hololink source directory
                         (default: /workspaces/cuda-qx/hololink)
  --cuda-quantum-dir DIR cuda-quantum source directory
                         (default: /workspaces/cuda-quantum)
  --jobs N               Parallel build jobs (default: nproc)

Network options:
  --device DEV           ConnectX IB device name (default: auto-detect)
  --bridge-ip ADDR       Bridge tool IP (default: 10.0.0.1)
  --emulator-ip ADDR     Emulator IP (default: 10.0.0.2)
  --fpga-ip ADDR         FPGA IP for non-emulate mode (default: 192.168.0.2)
  --mtu N                MTU size (default: 4096)

Run options:
  --gpu N                GPU device ID (default: 0)
  --timeout N            Timeout in seconds (default: 60)
  --no-verify            Skip ILA correction verification
  --num-shots N          Number of RPC messages (default: 100)
  --payload-size N       Bytes per RPC payload (default: 8)
  --page-size N          Ring buffer slot size in bytes (default: 384)
  --num-pages N          Number of ring buffer slots (default: 64)
  --control-port N       UDP control port for emulator (default: 8193)

  --help, -h             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --emulate)          EMULATE=true ;;
        --build)            DO_BUILD=true ;;
        --setup-network)    DO_SETUP_NETWORK=true ;;
        --no-run)           DO_RUN=false ;;
        --no-verify)        VERIFY=false ;;
        --hololink-dir)     HOLOLINK_DIR="$2"; shift ;;
        --cuda-quantum-dir) CUDA_QUANTUM_DIR="$2"; shift ;;
        --jobs)             JOBS="$2"; shift ;;
        --device)           IB_DEVICE="$2"; shift ;;
        --bridge-ip)        BRIDGE_IP="$2"; shift ;;
        --emulator-ip)      EMULATOR_IP="$2"; shift ;;
        --fpga-ip)          FPGA_IP="$2"; shift ;;
        --mtu)              MTU="$2"; shift ;;
        --gpu)              GPU_ID="$2"; shift ;;
        --timeout)          TIMEOUT="$2"; shift ;;
        --num-shots)        NUM_SHOTS="$2"; shift ;;
        --payload-size)     PAYLOAD_SIZE="$2"; shift ;;
        --page-size)        PAGE_SIZE="$2"; shift ;;
        --num-pages)        NUM_PAGES="$2"; shift ;;
        --control-port)     CONTROL_PORT="$2"; shift ;;
        --help|-h)          print_usage; exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
    shift
done

# ============================================================================
# Auto-detect IB device
# ============================================================================

detect_ib_device() {
    if [[ -n "$IB_DEVICE" ]]; then
        echo "$IB_DEVICE"
        return
    fi
    local dev
    dev=$(ibstat -l 2>/dev/null | head -1 || true)
    if [[ -z "$dev" ]]; then
        dev=$(ls /sys/class/infiniband/ 2>/dev/null | head -1 || true)
    fi
    if [[ -z "$dev" ]]; then
        echo "ERROR: Could not auto-detect IB device. Use --device." >&2
        exit 1
    fi
    echo "$dev"
}

# ============================================================================
# Network interface name from IB device
# ============================================================================

get_netdev() {
    local ib_dev=$1
    local netdev
    netdev=$(ls "/sys/class/infiniband/$ib_dev/device/net/" 2>/dev/null | head -1 || true)
    echo "$netdev"
}

# ============================================================================
# Build
# ============================================================================

do_build() {
    echo "=== Building tools ==="

    local realtime_dir="$CUDA_QUANTUM_DIR/realtime"
    local realtime_build="$realtime_dir/build"
    local hololink_build="$HOLOLINK_DIR/build"

    # Detect target arch
    local arch
    arch=$(uname -m)
    local target_arch="amd64"
    if [[ "$arch" == "aarch64" ]]; then
        target_arch="arm64"
    fi

    # Build hololink (only the two libraries we need)
    echo "--- Building hololink ($target_arch) ---"
    cmake -G Ninja -S "$HOLOLINK_DIR" -B "$hololink_build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTARGETARCH="$target_arch" \
        -DHOLOLINK_BUILD_ONLY_NATIVE=OFF \
        -DHOLOLINK_BUILD_PYTHON=OFF \
        -DHOLOLINK_BUILD_TESTS=OFF \
        -DHOLOLINK_BUILD_TOOLS=OFF \
        -DHOLOLINK_BUILD_EXAMPLES=OFF \
        -DHOLOLINK_BUILD_EMULATOR=OFF
    cmake --build "$hololink_build" -j"$JOBS" \
        --target gpu_roce_transceiver hololink_core

    # Build cuda-quantum/realtime with hololink tools enabled
    echo "--- Building cuda-quantum/realtime ---"
    cmake -G Ninja -S "$realtime_dir" -B "$realtime_build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS=ON \
        -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR="$HOLOLINK_DIR" \
        -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR="$hololink_build"
    cmake --build "$realtime_build" -j"$JOBS" \
        --target hololink_bridge hololink_fpga_emulator hololink_fpga_playback

    echo "=== Build complete ==="
}

# ============================================================================
# Network setup
# ============================================================================

do_setup_network() {
    IB_DEVICE=$(detect_ib_device)
    local netdev
    netdev=$(get_netdev "$IB_DEVICE")

    echo "=== Setting up network ==="
    echo "  IB device: $IB_DEVICE"
    echo "  Net device: $netdev"

    if [[ -z "$netdev" ]]; then
        echo "ERROR: No network device found for $IB_DEVICE" >&2
        exit 1
    fi

    sudo ip link set "$netdev" up mtu "$MTU" || true
    sudo ip addr add "$BRIDGE_IP/24" dev "$netdev" 2>/dev/null || true

    if $EMULATE; then
        sudo ip addr add "$EMULATOR_IP/24" dev "$netdev" 2>/dev/null || true
        # Add static ARP entries
        sudo ip neigh replace "$BRIDGE_IP" lladdr "$(cat /sys/class/net/$netdev/address)" dev "$netdev" nud permanent 2>/dev/null || true
        sudo ip neigh replace "$EMULATOR_IP" lladdr "$(cat /sys/class/net/$netdev/address)" dev "$netdev" nud permanent 2>/dev/null || true
    fi

    echo "=== Network setup complete ==="
}

# ============================================================================
# Run
# ============================================================================

cleanup_pids() {
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}

do_run() {
    IB_DEVICE=$(detect_ib_device)
    local build_dir="$CUDA_QUANTUM_DIR/realtime/build"
    local utils_dir="$build_dir/unittests/utils"

    local bridge_bin="$utils_dir/hololink_bridge"
    local emulator_bin="$utils_dir/hololink_fpga_emulator"
    local playback_bin="$utils_dir/hololink_fpga_playback"

    # Verify binaries exist
    for bin in "$bridge_bin"; do
        if [[ ! -x "$bin" ]]; then
            echo "ERROR: $bin not found. Run with --build first." >&2
            exit 1
        fi
    done

    PIDS=()
    trap cleanup_pids EXIT

    local FPGA_QP
    local FPGA_TARGET_IP

    if $EMULATE; then
        echo "=== Emulated mode ==="

        # Start emulator
        echo "--- Starting emulator ---"
        "$emulator_bin" \
            --device="$IB_DEVICE" \
            --port="$CONTROL_PORT" \
            --bridge-ip="$BRIDGE_IP" \
            --page-size="$PAGE_SIZE" \
            2>&1 | tee /tmp/emulator.log &
        PIDS+=($!)

        # Wait for emulator to print QP number
        sleep 2
        FPGA_QP=$(grep -oP 'QP Number: 0x\K[0-9a-fA-F]+' /tmp/emulator.log | head -1)
        if [[ -z "$FPGA_QP" ]]; then
            echo "ERROR: Could not parse emulator QP from log" >&2
            exit 1
        fi
        FPGA_QP="0x$FPGA_QP"
        FPGA_TARGET_IP="$EMULATOR_IP"

        echo "  Emulator QP: $FPGA_QP"
    else
        echo "=== FPGA mode ==="
        FPGA_QP="0x2"
        FPGA_TARGET_IP="$FPGA_IP"
    fi

    # Start bridge
    echo "--- Starting bridge ---"
    "$bridge_bin" \
        --device="$IB_DEVICE" \
        --peer-ip="$FPGA_TARGET_IP" \
        --remote-qp="$FPGA_QP" \
        --gpu="$GPU_ID" \
        --timeout="$TIMEOUT" \
        --page-size="$PAGE_SIZE" \
        --num-pages="$NUM_PAGES" \
        2>&1 | tee /tmp/bridge.log &
    PIDS+=($!)

    # Wait for bridge to print QP info
    sleep 3
    local BRIDGE_QP BRIDGE_RKEY BRIDGE_BUFFER
    BRIDGE_QP=$(grep -oP 'QP Number: 0x\K[0-9a-fA-F]+' /tmp/bridge.log | tail -1)
    BRIDGE_RKEY=$(grep -oP 'RKey: \K[0-9]+' /tmp/bridge.log | tail -1)
    BRIDGE_BUFFER=$(grep -oP 'Buffer Addr: 0x\K[0-9a-fA-F]+' /tmp/bridge.log | tail -1)

    if [[ -z "$BRIDGE_QP" || -z "$BRIDGE_RKEY" || -z "$BRIDGE_BUFFER" ]]; then
        echo "ERROR: Could not parse bridge QP info from log" >&2
        echo "  QP=$BRIDGE_QP RKEY=$BRIDGE_RKEY BUFFER=$BRIDGE_BUFFER" >&2
        exit 1
    fi

    echo "  Bridge QP: 0x$BRIDGE_QP"
    echo "  Bridge RKey: $BRIDGE_RKEY"
    echo "  Bridge Buffer: 0x$BRIDGE_BUFFER"

    # Start playback
    echo "--- Starting playback ---"
    local verify_flag=""
    if ! $VERIFY; then
        verify_flag="--no-verify"
    fi

    "$playback_bin" \
        --control-ip="$FPGA_TARGET_IP" \
        --control-port="$CONTROL_PORT" \
        --bridge-qp="0x$BRIDGE_QP" \
        --bridge-rkey="$BRIDGE_RKEY" \
        --bridge-buffer="0x$BRIDGE_BUFFER" \
        --page-size="$PAGE_SIZE" \
        --num-pages="$NUM_PAGES" \
        --num-shots="$NUM_SHOTS" \
        --payload-size="$PAYLOAD_SIZE" \
        --bridge-ip="$BRIDGE_IP" \
        $verify_flag
    PLAYBACK_EXIT=$?

    # Wait for bridge to finish
    sleep 2

    # Cleanup
    cleanup_pids

    echo ""
    if [[ $PLAYBACK_EXIT -eq 0 ]]; then
        echo "*** TEST PASSED ***"
    else
        echo "*** TEST FAILED ***"
    fi
    exit $PLAYBACK_EXIT
}

# ============================================================================
# Main
# ============================================================================

echo "=== Hololink Generic RPC Test ==="
echo "Mode: $(if $EMULATE; then echo "emulated"; else echo "FPGA"; fi)"

if $DO_BUILD; then
    do_build
fi

if $DO_SETUP_NETWORK; then
    do_setup_network
fi

if $DO_RUN; then
    do_run
fi

echo "Done."
