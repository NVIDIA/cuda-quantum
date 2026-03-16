#!/bin/bash
# ============================================================================#
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================#
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
HOLOLINK_DIR="/workspaces/hololink"
CUDA_QUANTUM_DIR="/workspaces/cuda-quantum"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR=""

# Network defaults
IB_DEVICE=""           # auto-detect
BRIDGE_IP="10.0.0.1"
EMULATOR_IP="10.0.0.2"
FPGA_IP="192.168.0.2"
MTU=4096

# Run defaults
GPU_ID=0
TIMEOUT=60
NUM_MESSAGES=100
PAYLOAD_SIZE=8
PAGE_SIZE=384
NUM_PAGES=128
CONTROL_PORT=8193
FORWARD=false
UNIFIED=false

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
                         (default: /workspaces/hololink)
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
  --no-verify            Skip ILA response verification
  --num-messages N       Number of RPC messages (default: 100)
  --payload-size N       Bytes per RPC payload (default: 16)
  --page-size N          Ring buffer slot size in bytes (default: 384)
  --num-pages N          Number of ring buffer slots (default: 128)
  --control-port N       UDP control port for emulator (default: 8193)
  --bin-dir DIR          Binary directory containing executables (default: None)
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
        --bin-dir)          BIN_DIR="$2"; shift ;;
        --jobs)             JOBS="$2"; shift ;;
        --device)           IB_DEVICE="$2"; shift ;;
        --bridge-ip)        BRIDGE_IP="$2"; shift ;;
        --emulator-ip)      EMULATOR_IP="$2"; shift ;;
        --fpga-ip)          FPGA_IP="$2"; shift ;;
        --mtu)              MTU="$2"; shift ;;
        --gpu)              GPU_ID="$2"; shift ;;
        --forward)          FORWARD=true ;;
        --unified)          UNIFIED=true ;;
        --timeout)          TIMEOUT="$2"; shift ;;
        --num-messages)     NUM_MESSAGES="$2"; shift ;;
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

detect_cuda_arch() {
    local max_arch
    max_arch=$(nvcc --list-gpu-arch 2>/dev/null \
        | grep -oP 'compute_\K[0-9]+' | sort -n | tail -1)
    if [ -n "$max_arch" ]; then
        echo "$max_arch"
    fi
}

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

    # Detect highest CUDA arch supported by nvcc
    local cuda_arch
    cuda_arch=$(detect_cuda_arch)
    local cuda_arch_flag=""
    if [ -n "$cuda_arch" ]; then
        cuda_arch_flag="-DCMAKE_CUDA_ARCHITECTURES=$cuda_arch"
        echo "  CUDA arch: $cuda_arch"
    fi

    # Build hololink (only the two libraries we need)
    echo "--- Building hololink ($target_arch) ---"
    cmake -G Ninja -S "$HOLOLINK_DIR" -B "$hololink_build" \
        -DCMAKE_BUILD_TYPE=Release \
        $cuda_arch_flag \
        -DTARGET_ARCH="$target_arch" \
        -DHOLOLINK_BUILD_ONLY_NATIVE=OFF \
        -DHOLOLINK_BUILD_PYTHON=OFF \
        -DHOLOLINK_BUILD_TESTS=OFF \
        -DHOLOLINK_BUILD_TOOLS=OFF \
        -DHOLOLINK_BUILD_EXAMPLES=OFF \
        -DHOLOLINK_BUILD_EMULATOR=OFF
    cmake --build "$hololink_build" -j"$JOBS" \
        --target roce_receiver gpu_roce_transceiver hololink_core

    # Build cuda-quantum/realtime with hololink tools enabled
    echo "--- Building cuda-quantum/realtime ---"
    cmake -G Ninja -S "$realtime_dir" -B "$realtime_build" \
        -DCMAKE_BUILD_TYPE=Release \
        $cuda_arch_flag \
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

setup_port() {
    local iface="$1"
    local ip="$2"
    local mtu="$3"

    echo "  Configuring $iface: ip=$ip mtu=$mtu"

    # Remove this IP from any other interface to prevent routing ambiguity
    local other
    for other in $(ip -o addr show to "${ip}/24" 2>/dev/null | awk '{print $2}' | sort -u); do
        if [[ "$other" != "$iface" ]]; then
            echo "    Removing stale ${ip}/24 from $other"
            sudo ip addr del "${ip}/24" dev "$other" 2>/dev/null || true
        fi
    done

    sudo ip link set "$iface" up
    sudo ip link set "$iface" mtu "$mtu"
    sudo ip addr flush dev "$iface"
    sudo ip addr add "${ip}/24" dev "$iface"

    # Configure RoCEv2 mode
    local ib_dev
    if command -v ibdev2netdev &>/dev/null; then
        ib_dev=$(ibdev2netdev | awk -v iface="$iface" '$5 == iface { print $1 }')
    fi
    if [[ -z "$ib_dev" ]]; then
        ib_dev=$(basename "$(ls -d /sys/class/net/$iface/device/infiniband/* 2>/dev/null | head -1)" 2>/dev/null || true)
    fi
    if [[ -n "$ib_dev" ]]; then
        local has_rocev2=false
        for f in /sys/class/infiniband/${ib_dev}/ports/*/gid_attrs/types/*; do
            if [[ -f "$f" ]] && grep -q "RoCE v2" "$f" 2>/dev/null; then
                has_rocev2=true; break
            fi
        done
        if $has_rocev2; then
            echo "    RoCEv2 GID available for $ib_dev"
        elif command -v rdma &>/dev/null && rdma link set --help &>/dev/null; then
            local port_count
            port_count=$(ls -d "/sys/class/infiniband/${ib_dev}/ports/"* 2>/dev/null | wc -l)
            for p in $(seq 1 "$port_count"); do
                sudo rdma link set "${ib_dev}/${p}" type eth || true
            done
            echo "    RoCEv2 mode configured for $ib_dev"
        else
            echo "    WARNING: Could not verify RoCEv2 mode for $ib_dev"
        fi
    fi

    # DSCP trust mode for lossless RoCE
    if command -v mlnx_qos &>/dev/null; then
        sudo mlnx_qos -i "$iface" --trust=dscp 2>/dev/null || true
        echo "    DSCP trust mode set"
    fi

    # Disable adaptive RX coalescing for low latency
    if command -v ethtool &>/dev/null; then
        sudo ethtool -C "$iface" adaptive-rx off rx-usecs 0 2>/dev/null || true
    fi

    echo "    Done: $iface is up at $ip"
}

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

    if $EMULATE; then
        setup_port "$netdev" "$BRIDGE_IP" "$MTU"
        sudo ip addr add "$EMULATOR_IP/24" dev "$netdev" 2>/dev/null || true
        # Add static ARP entries for loopback
        local mac
        mac=$(cat /sys/class/net/$netdev/address)
        sudo ip neigh replace "$BRIDGE_IP" lladdr "$mac" dev "$netdev" nud permanent 2>/dev/null || true
        sudo ip neigh replace "$EMULATOR_IP" lladdr "$mac" dev "$netdev" nud permanent 2>/dev/null || true
    else
        setup_port "$netdev" "$BRIDGE_IP" "$MTU"
    fi

    echo "=== Network setup complete ==="
}

# ============================================================================
# Run
# ============================================================================

cleanup_pids() {
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -INT "$pid" 2>/dev/null || true
        fi
    done
    # Give processes a moment to shut down gracefully
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}

do_run() {
    IB_DEVICE=$(detect_ib_device)
    local build_dir="$CUDA_QUANTUM_DIR/realtime/build"
    local utils_dir="$build_dir/unittests/utils"

    if [ -n "$BIN_DIR" ]; then
        local bridge_bin="$BIN_DIR/hololink_bridge"
        local emulator_bin="$BIN_DIR/hololink_fpga_emulator"
        local playback_bin="$BIN_DIR/hololink_fpga_playback"
    else
        local bridge_bin="$utils_dir/hololink_bridge"
        local emulator_bin="$utils_dir/hololink_fpga_emulator"
        local playback_bin="$utils_dir/hololink_fpga_playback"
    fi

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
        > /tmp/emulator.log
        "$emulator_bin" \
            --device="$IB_DEVICE" \
            --port="$CONTROL_PORT" \
            --bridge-ip="$BRIDGE_IP" \
            --page-size="$PAGE_SIZE" \
            > /tmp/emulator.log 2>&1 &
        PIDS+=($!)
        tail -f /tmp/emulator.log &
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
    > /tmp/bridge.log
    local bridge_args=(
        --device="$IB_DEVICE"
        --peer-ip="$FPGA_TARGET_IP"
        --remote-qp="$FPGA_QP"
        --gpu="$GPU_ID"
        --timeout="$TIMEOUT"
        --payload-size="$PAYLOAD_SIZE"
        --page-size="$PAGE_SIZE"
        --num-pages="$NUM_PAGES"
    )
    if $FORWARD; then
        bridge_args+=(--forward)
    fi
    if $UNIFIED; then
        bridge_args+=(--unified)
    fi
    CUDA_MODULE_LOADING=EAGER "$bridge_bin" "${bridge_args[@]}" > /tmp/bridge.log 2>&1 &
    BRIDGE_PID=$!
    PIDS+=($BRIDGE_PID)
    tail -f /tmp/bridge.log &
    PIDS+=($!)

    # Wait for bridge to print "Bridge Ready" (CUDA/DOCA init can take 5-15s)
    local wait_elapsed=0
    while ! grep -q "Bridge Ready" /tmp/bridge.log 2>/dev/null; do
        if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
            echo "ERROR: Bridge process died during startup" >&2
            cat /tmp/bridge.log >&2
            exit 1
        fi
        if (( wait_elapsed >= 30 )); then
            echo "ERROR: Bridge did not become ready within 30s" >&2
            cat /tmp/bridge.log >&2
            exit 1
        fi
        sleep 1
        (( wait_elapsed++ )) || true
    done

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
    local playback_args=(
        --hololink="$FPGA_TARGET_IP"
        --bridge-qp="0x$BRIDGE_QP"
        --bridge-rkey="$BRIDGE_RKEY"
        --bridge-buffer="0x$BRIDGE_BUFFER"
        --page-size="$PAGE_SIZE"
        --num-pages="$NUM_PAGES"
        --num-messages="$NUM_MESSAGES"
        --payload-size="$PAYLOAD_SIZE"
        --bridge-ip="$BRIDGE_IP"
    )
    if $EMULATE; then
        playback_args+=(--emulator --control-port="$CONTROL_PORT")
    fi
    if ! $VERIFY; then
        playback_args+=(--no-verify)
    fi
    if $FORWARD; then
        playback_args+=(--forward)
    fi

    "$playback_bin" "${playback_args[@]}"
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

if [ -n "$BIN_DIR" ]; then
   if $DO_BUILD; then
    echo "Cannot request a build when the binary directory is provided."
   fi 
fi

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
