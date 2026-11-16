#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#
# roce_test.sh
#
# Shared orchestration for CPU/GPU RoCE RPC validation.
# Tests libcudaq-realtime host/device dispatch through the bridge API with a
# simple increment RPC handler (no QEC or decoder dependency).
# The receiver loads a bridge plugin; this runner drives hsb_fpga_playback
# directly. validation_test.sh handles the playback-provider plugin workflow.
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
#   ./roce_test.sh --emulate --build --setup-network
#
#   # Just run with real FPGA (tools already built, network already set up)
#   ./roce_test.sh --fpga-ip 192.168.0.2
#
#   # Build only
#   ./roce_test.sh --build --no-run
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
HSB_DIR="/workspaces/hololink"
CUDA_QUANTUM_DIR="/workspaces/cuda-quantum"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR=""
# The selected app owns bridge setup; only validation_app accepts opaque
# bridge options. roce_app instead translates the common RoCE settings.
APP=roce_app
BRIDGE_LIBRARY=""
BRIDGE_ARGS=()

# Network defaults
IB_DEVICE=""           # auto-detect
BRIDGE_IP="10.0.0.1"
EMULATOR_IP="10.0.0.2"
FPGA_IP="192.168.0.2"
MTU=4096

# Run defaults
BACKEND=""
GPU_ID=0
SAVE_TEMPS=false
TIMEOUT=60
NUM_MESSAGES=100
PAYLOAD_SIZE=8
PAGE_SIZE=384
NUM_PAGES=128
CONTROL_PORT=8193
FORWARD=false
UNIFIED=false
DISPATCHER="builtin"
BLOCKS=0
BLOCKS_SET=false
THREADS=32
# Build parallelism
JOBS=$(nproc 2>/dev/null || echo 8)

# ============================================================================
# Argument Parsing
# ============================================================================

print_usage() {
    cat <<'EOF'
Usage: roce_test.sh [options]

Modes:
  --emulate              Use FPGA emulator (3-tool mode, no FPGA needed)
                         Default: FPGA mode (2-tool, requires real FPGA)

Actions:
  --build                Build all required tools before running
  --setup-network        Configure ConnectX network interfaces
  --no-run               Skip running the test (useful with --build)

Build options:
  --hsb-dir DIR     HSB source directory
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
  --forward              Echo payloads without RPC dispatch
  --unified              Combine RX, RPC dispatch, and TX in one thread/kernel
                         Default: separate RX, dispatch, and TX threads/kernels
  --cpu                  CPU RoCE bridge and HOST_CALL (no GPU execution)
  --gpu [N]              GPU RoCE bridge and DEVICE_CALL (default; optional ID)
  --gpu-id N             GPU device ID (default: 0)
  --save-temps           Keep receiver/emulator logs in a temporary directory
  --timeout N            Timeout in seconds (default: 60)
  --no-verify            Skip ILA response verification
  --num-messages N       Number of RPC messages (default: 100)
  --payload-size N       Bytes per RPC payload (default: 8)
  --page-size N          Ring buffer slot size in bytes (default: 384)
  --num-pages N          Number of ring buffer slots (default: 128)
  --dispatcher NAME      Unified GPU dispatcher: builtin or grid (default: builtin)
  --blocks N             Grid blocks; 0 uses the SM count (default: 0)
  --threads N            Dispatch block size; every DEVICE_CALL handler runs
                         on it, in both dispatch modes (default: 32)
  --control-port N       UDP control port for emulator (default: 8193)
  --app NAME             roce_app (default) or validation_app
  --bridge LIB           Provider library for validation_app
  --                     Forward all remaining arguments unchanged to the bridge
  --bin-dir DIR          Binary directory containing executables (default: None)
  --help, -h             Show this help

Dependencies:
  HSB playback requires the CUDA driver library (libcuda.so.1), including in
  CPU mode. CPU RoCE also requires libibverbs and a configured RDMA device.
EOF
}

select_backend() {
    if [[ -n "$BACKEND" && "$BACKEND" != "$1" ]]; then
        echo "ERROR: --cpu and --gpu are mutually exclusive" >&2
        exit 1
    fi
    BACKEND="$1"
}

# This direct-tool parser accepts separate option/value tokens. The generic
# runner normalizes its compatibility options before delegating here; tokens
# after -- remain untouched for validation_app's bridge.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --emulate)          EMULATE=true ;;
        --build)            DO_BUILD=true ;;
        --setup-network)    DO_SETUP_NETWORK=true ;;
        --no-run)           DO_RUN=false ;;
        --no-verify)        VERIFY=false ;;
        --hsb-dir)     HSB_DIR="$2"; shift ;;
        --cuda-quantum-dir) CUDA_QUANTUM_DIR="$2"; shift ;;
        --bin-dir)          BIN_DIR="$2"; shift ;;
        --app)              APP="$2"; shift ;;
        --bridge)           BRIDGE_LIBRARY="$2"; shift ;;
        --)                 shift; BRIDGE_ARGS=("$@"); break ;;
        --jobs)             JOBS="$2"; shift ;;
        --device)           IB_DEVICE="$2"; shift ;;
        --bridge-ip)        BRIDGE_IP="$2"; shift ;;
        --emulator-ip)      EMULATOR_IP="$2"; shift ;;
        --fpga-ip)          FPGA_IP="$2"; shift ;;
        --mtu)              MTU="$2"; shift ;;
        --cpu)              select_backend cpu ;;
        --gpu)              select_backend gpu
                            if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
                                GPU_ID="$2"; shift
                            fi ;;
        --gpu-id)           GPU_ID="$2"; shift ;;
        --save-temps)       SAVE_TEMPS=true ;;
        --forward)          FORWARD=true ;;
        --unified)          UNIFIED=true ;;
        --dispatcher)       DISPATCHER="$2"; shift ;;
        --dispatcher=*)     DISPATCHER="${1#*=}" ;;
        --blocks)           BLOCKS="$2"; BLOCKS_SET=true; shift ;;
        --blocks=*)         BLOCKS="${1#*=}"; BLOCKS_SET=true ;;
        --threads)          THREADS="$2"; shift ;;
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
BACKEND="${BACKEND:-gpu}"
case "$APP" in
    roce_app)
        if [[ -n "$BRIDGE_LIBRARY" || ${#BRIDGE_ARGS[@]} -ne 0 ]]; then
            echo "ERROR: --bridge and passthrough arguments require --app validation_app" >&2
            exit 1
        fi ;;
    validation_app)
        if [[ -z "$BRIDGE_LIBRARY" ]]; then
            echo "ERROR: --app validation_app requires --bridge LIB" >&2
            exit 1
        fi ;;
    *) echo "ERROR: --app must be roce_app or validation_app" >&2; exit 1 ;;
esac
if $FORWARD && $UNIFIED; then
    echo "ERROR: --forward and --unified are mutually exclusive" >&2
    exit 1
fi
if [[ "$DISPATCHER" != "builtin" && "$DISPATCHER" != "grid" ]]; then
    echo "ERROR: --dispatcher must be builtin or grid" >&2
    exit 1
fi
if [[ "$DISPATCHER" == "grid" ]]; then
    if ! $UNIFIED || [[ "$BACKEND" != gpu || "$APP" != roce_app ]]; then
        echo "ERROR: --dispatcher grid requires --unified, --gpu and --app roce_app" >&2
        exit 1
    fi
fi
if $BLOCKS_SET && [[ "$DISPATCHER" != "grid" ]]; then
    echo "ERROR: --blocks requires --dispatcher grid" >&2
    exit 1
fi
if ! [[ "$BLOCKS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --blocks must be a non-negative integer" >&2
    exit 1
fi

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
    local hsb_build="$HSB_DIR/build"

    # Detect target arch
    local arch
    arch=$(uname -m)
    local target_arch="amd64"
    if [[ "$arch" == "aarch64" ]]; then
        target_arch="arm64"
    fi

    # Ensure nvcc is on PATH for detect_cuda_arch() and cmake check_language(CUDA).
    if [[ -x /usr/local/cuda/bin/nvcc ]]; then
        case ":$PATH:" in
            *":/usr/local/cuda/bin:"*) ;;
            *) export PATH="/usr/local/cuda/bin:$PATH" ;;
        esac
    fi

    # Detect highest CUDA arch supported by nvcc
    local cuda_arch
    cuda_arch=$(detect_cuda_arch)
    local cuda_arch_flag=""
    if [ -n "$cuda_arch" ]; then
        cuda_arch_flag="-DCMAKE_CUDA_ARCHITECTURES=$cuda_arch"
        echo "  CUDA arch: $cuda_arch"
    fi

    # Build HSB (only the two libraries we need)
    echo "--- Building HSB ($target_arch) ---"
    cmake -G Ninja -S "$HSB_DIR" -B "$hsb_build" \
        -DCMAKE_BUILD_TYPE=Release \
        $cuda_arch_flag \
        -DTARGET_ARCH="$target_arch" \
        -DHOLOLINK_BUILD_ONLY_NATIVE=OFF \
        -DHOLOLINK_BUILD_PYTHON=OFF \
        -DHOLOLINK_BUILD_TESTS=OFF \
        -DHOLOLINK_BUILD_TOOLS=OFF \
        -DHOLOLINK_BUILD_EXAMPLES=OFF \
        -DHOLOLINK_BUILD_EMULATOR=OFF
    cmake --build "$hsb_build" -j"$JOBS" \
        --target roce_receiver gpu_roce_transceiver hololink_core

    # Build cuda-quantum/realtime with HSB tools enabled
    echo "--- Building cuda-quantum/realtime ---"
    cmake -G Ninja -S "$realtime_dir" -B "$realtime_build" \
        -DCMAKE_BUILD_TYPE=Release \
        $cuda_arch_flag \
        -DCUDAQ_REALTIME_ENABLE_HSB_TOOLS=ON \
        -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR="$HSB_DIR" \
        -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR="$hsb_build"
    cmake --build "$realtime_build" -j"$JOBS" \
        --target roce_app validation_app hsb_fpga_emulator hsb_fpga_playback

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
    local ib_dev=""
    if command -v ibdev2netdev &>/dev/null; then
        ib_dev=$(ibdev2netdev | awk -v iface="$iface" '$5 == iface { print $1 }')
    fi
    if [[ -z "$ib_dev" ]]; then
        ib_dev=$(basename "$(ls -d /sys/class/net/$iface/device/infiniband/* 2>/dev/null | head -1)" 2>/dev/null || true)
    fi
    if [[ -n "$ib_dev" ]]; then
        local has_rocev2=false
        for gid_type_file in /sys/class/infiniband/${ib_dev}/ports/*/gid_attrs/types/*; do
            if [[ -f "$gid_type_file" ]] && grep -q "RoCE v2" "$gid_type_file" 2>/dev/null; then
                has_rocev2=true; break
            fi
        done
        if $has_rocev2; then
            echo "    RoCEv2 GID available for $ib_dev"
        elif command -v rdma &>/dev/null && rdma link set --help &>/dev/null; then
            local port_count
            port_count=$(ls -d "/sys/class/infiniband/${ib_dev}/ports/"* 2>/dev/null | wc -l)
            for port_number in $(seq 1 "$port_count"); do
                sudo rdma link set "${ib_dev}/${port_number}" type eth || true
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
    PIDS=()
}

cleanup() {
    cleanup_pids
    if $SAVE_TEMPS; then
        echo "Saved logs: $LOG_DIR"
    else
        rm -rf -- "$LOG_DIR"
    fi
}

do_run() {
    IB_DEVICE=$(detect_ib_device)
    local build_dir="$CUDA_QUANTUM_DIR/realtime/build"
    local validation_dir="$build_dir/validation"

    if [ -n "$BIN_DIR" ]; then
        local bridge_bin="$BIN_DIR/$APP"
        local emulator_bin="$BIN_DIR/hsb_fpga_emulator"
        local playback_bin="$BIN_DIR/hsb_fpga_playback"
        if [[ ! -x "$bridge_bin" ]]; then
            bridge_bin="$BIN_DIR/../validation/$APP"
        fi
    else
        local bridge_bin="$validation_dir/$APP"
        local emulator_bin="$validation_dir/hsb_fpga_emulator"
        local playback_bin="$validation_dir/hsb_fpga_playback"
    fi

    # All modes use the bridge API app. connect() prints QP information before
    # launch(), so wait until both the dispatcher and transport are ready.
    local ready_marker="=== App Ready ==="

    # Verify binaries exist
    for bin in "$bridge_bin"; do
        if [[ ! -x "$bin" ]]; then
            echo "ERROR: $bin not found. Run with --build first." >&2
            exit 1
        fi
    done

    PIDS=()
    LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/roce-validation.XXXXXX")
    BRIDGE_LOG="$LOG_DIR/bridge.log"
    EMULATOR_LOG="$LOG_DIR/emulator.log"
    echo "Logs: $LOG_DIR"
    trap cleanup EXIT

    local FPGA_QP
    local FPGA_TARGET_IP

    if $EMULATE; then
        echo "=== Emulated mode ==="

        # Start emulator
        echo "--- Starting emulator ---"
        > "$EMULATOR_LOG"
        "$emulator_bin" \
            --device="$IB_DEVICE" \
            --port="$CONTROL_PORT" \
            --bridge-ip="$BRIDGE_IP" \
            --page-size="$PAGE_SIZE" \
            > "$EMULATOR_LOG" 2>&1 &
        PIDS+=($!)
        tail -f "$EMULATOR_LOG" &
        PIDS+=($!)

        # Wait for emulator to print QP number
        sleep 2
        FPGA_QP=$(grep -oP 'QP Number: 0x\K[0-9a-fA-F]+' "$EMULATOR_LOG" | head -1)
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
    > "$BRIDGE_LOG"
    local bridge_args=(
        --"$BACKEND"
        --gpu-id="$GPU_ID"
        --timeout="$TIMEOUT"
        --threads="$THREADS"
    )
    # roce_app translates these transport settings into bridge options itself.
    # validation_app instead takes the caller's explicit library and argv below.
    if [[ "$APP" == roce_app ]]; then
        bridge_args+=(
            --device="$IB_DEVICE"
            --peer-ip="$FPGA_TARGET_IP"
            --remote-qp="$FPGA_QP"
            --payload-size="$PAYLOAD_SIZE"
            --page-size="$PAGE_SIZE"
            --num-pages="$NUM_PAGES"
        )
    fi
    if $FORWARD; then
        bridge_args+=(--forward)
    fi
    if $UNIFIED; then
        bridge_args+=(--unified)
        if [[ "$DISPATCHER" == "grid" ]]; then
            bridge_args+=(--dispatcher="$DISPATCHER" --blocks="$BLOCKS")
        fi
    fi
    if [[ "$APP" == validation_app ]]; then
        bridge_args+=(--bridge="$BRIDGE_LIBRARY" -- "${BRIDGE_ARGS[@]}")
    fi
    CUDA_MODULE_LOADING=EAGER "$bridge_bin" "${bridge_args[@]}" > "$BRIDGE_LOG" 2>&1 &
    BRIDGE_PID=$!
    PIDS+=($BRIDGE_PID)
    tail -f "$BRIDGE_LOG" &
    PIDS+=($!)

    # Wait for the app to finish dispatcher and bridge startup.
    local wait_elapsed=0
    while ! grep -q "$ready_marker" "$BRIDGE_LOG" 2>/dev/null; do
        if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
            echo "ERROR: Bridge process died during startup" >&2
            cat "$BRIDGE_LOG" >&2
            exit 1
        fi
        if (( wait_elapsed >= 30 )); then
            echo "ERROR: Bridge did not become ready within 30s" >&2
            cat "$BRIDGE_LOG" >&2
            exit 1
        fi
        sleep 1
        (( wait_elapsed++ )) || true
    done

    # Direct HSB playback needs these destination values as separate options;
    # the generic provider runner instead forwards one opaque endpoint string.
    local BRIDGE_QP BRIDGE_RKEY BRIDGE_BUFFER
    BRIDGE_QP=$(grep -oP 'QP Number: 0x\K[0-9a-fA-F]+' "$BRIDGE_LOG" | tail -1)
    BRIDGE_RKEY=$(grep -oP 'RKey: \K[0-9]+' "$BRIDGE_LOG" | tail -1)
    BRIDGE_BUFFER=$(grep -oP 'Buffer Addr: 0x\K[0-9a-fA-F]+' "$BRIDGE_LOG" | tail -1)

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
        --hsb-ip="$FPGA_TARGET_IP"
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

    PLAYBACK_EXIT=0
    "$playback_bin" "${playback_args[@]}" || PLAYBACK_EXIT=$?

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

echo "=== RoCE RPC Validation ($BACKEND) ==="
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
