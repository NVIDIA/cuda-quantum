#!/bin/bash
# ============================================================================#
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================#
#
# gpu_grid_dispatch_test.sh
#
# End-to-end run of the gpu_grid_dispatch example: the HSB FPGA (or its
# emulator) sends rpc_increment requests over RoCE, gpu_grid_dispatch receives
# them on the CPU through the cpu_roce bridge, hands each to its resident GPU
# grid and sends the reply, and hsb_fpga_playback checks the replies.
# Modelled on unittests/cpu_transport/hsb_test_cpu.sh, which runs the same
# wire protocol against hsb_bridge_cpu.
#
# Modes:
#   Default (FPGA):   gpu_grid_dispatch + playback           (real FPGA)
#   --emulate:        emulator + gpu_grid_dispatch + playback (no FPGA)
#
# The emulator and playback tools come from a CUDA-Q Realtime build with the
# HSB tools enabled (CUDAQ_REALTIME_ENABLE_HSB_TOOLS=ON); point --bin-dir at
# its unittests/utils directory.  Network setup is not repeated here: run
# hsb_test_cpu.sh --setup-network (same addresses) first.
#
# Examples:
#   ./gpu_grid_dispatch_test.sh --emulate --bin-dir /path/to/build/unittests/utils
#   ./gpu_grid_dispatch_test.sh --fpga-ip 192.168.0.2 --notify pinned --poll leader
#   ./gpu_grid_dispatch_test.sh --emulate --request ring \
#       --page-size 4096 --payload-size 1000 --num-messages 32
set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================

EMULATE=false
VERIFY=true
SAVE_TEMPS=false

EMULATOR_LOG=""
SERVER_LOG=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_QUANTUM_DIR="/workspaces/cuda-quantum"
BIN_DIR=""
EXAMPLE_BIN=""
BRIDGE_LIB=""

# Network defaults (match hsb_test_cpu.sh so the wiring is interchangeable).
IB_DEVICE=""           # auto-detect
BRIDGE_IP="10.0.0.1"
EMULATOR_IP="10.0.0.2"
FPGA_IP="192.168.0.2"
FPGA_QP="0x2"

# Run defaults
GPU_ID=0
NOTIFY="auto"
POLL="auto"
REQUEST="inline"
BLOCKS=""
THREADS=""
TIMEOUT=60
NUM_MESSAGES=100
PAYLOAD_SIZE=8
PAGE_SIZE=384
NUM_PAGES=64           # the HSB WQE depth; cpu_roce clamps to it
FRAME_SIZE=""
CONTROL_PORT=8193

# ============================================================================
# Argument parsing
# ============================================================================
print_usage() {
    cat <<'EOF_USAGE'
Usage: gpu_grid_dispatch_test.sh [options]

Modes:
  --emulate              Use the FPGA emulator (no FPGA needed)

Binaries:
  --bin-dir DIR          Directory with hsb_fpga_emulator and hsb_fpga_playback
                         (default: <cuda-quantum-dir>/realtime/build/unittests/utils)
  --cuda-quantum-dir DIR cuda-quantum source dir (default: /workspaces/cuda-quantum)
  --example-bin PATH     gpu_grid_dispatch binary
                         (default: <script dir>/build/gpu_grid_dispatch, then
                         <script dir>/gpu_grid_dispatch)
  --bridge PATH          cpu_roce bridge provider .so (default: the example's)

Network options:
  --device DEV           ConnectX IB device name (default: auto-detect)
  --bridge-ip ADDR       This host's RoCE IP (default: 10.0.0.1)
  --emulator-ip ADDR     Emulator IP (default: 10.0.0.2)
  --fpga-ip ADDR         FPGA IP for non-emulate mode (default: 192.168.0.2)
  --fpga-qp QP           FPGA data-plane QP (default: 0x2)

Example options (passed to gpu_grid_dispatch):
  --gpu N                CUDA device (default: 0)
  --notify MODE          auto | gdrcopy | pinned (default: auto)
  --poll MODE            auto | all | leader (default: auto)
  --request MODE         inline | ring (default: inline); ring serves
                         payloads up to page size - 24 bytes
  --blocks N             grid blocks (default: SM count)
  --threads N            threads per block (default: 256)

Run options:
  --timeout N            Server timeout in seconds (default: 60)
  --no-verify            Skip response verification
  --num-messages N       Number of RPC messages (default: 100)
  --payload-size N       Bytes per RPC payload (default: 8); inline mode
                         answers the first 8, ring mode all of them.  The
                         playback BRAM holds 512 64-byte lines in all, so
                         num-messages x ceil((24 + N) / 64) <= 512, and the
                         response capture checks the first 40 bytes
  --page-size N          Ring slot size in bytes (default: 384); raise it
                         with --payload-size (24 + payload must fit)
  --num-pages N          Number of ring slots (default: 64)
  --frame-size N         Response bytes on the wire (default: slot size)
  --control-port N       UDP control port for emulator (default: 8193)
  --save-temps           Keep the log files (paths are printed)
  --help, -h             Show this help
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --emulate)          EMULATE=true ;;
        --no-verify)        VERIFY=false ;;
        --save-temps)       SAVE_TEMPS=true ;;
        --bin-dir)          BIN_DIR="$2"; shift ;;
        --cuda-quantum-dir) CUDA_QUANTUM_DIR="$2"; shift ;;
        --example-bin)      EXAMPLE_BIN="$2"; shift ;;
        --bridge)           BRIDGE_LIB="$2"; shift ;;
        --device)           IB_DEVICE="$2"; shift ;;
        --bridge-ip)        BRIDGE_IP="$2"; shift ;;
        --emulator-ip)      EMULATOR_IP="$2"; shift ;;
        --fpga-ip)          FPGA_IP="$2"; shift ;;
        --fpga-qp)          FPGA_QP="$2"; shift ;;
        --gpu)              GPU_ID="$2"; shift ;;
        --notify)           NOTIFY="$2"; shift ;;
        --poll)             POLL="$2"; shift ;;
        --request)          REQUEST="$2"; shift ;;
        --blocks)           BLOCKS="$2"; shift ;;
        --threads)          THREADS="$2"; shift ;;
        --timeout)          TIMEOUT="$2"; shift ;;
        --num-messages)     NUM_MESSAGES="$2"; shift ;;
        --payload-size)     PAYLOAD_SIZE="$2"; shift ;;
        --page-size)        PAGE_SIZE="$2"; shift ;;
        --num-pages)        NUM_PAGES="$2"; shift ;;
        --frame-size)       FRAME_SIZE="$2"; shift ;;
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

# Catch sizes the tools would reject later, with the reason.
if (( PAYLOAD_SIZE + 24 > PAGE_SIZE )); then
    echo "ERROR: --payload-size $PAYLOAD_SIZE + the 24-byte header exceeds" \
         "--page-size $PAGE_SIZE; raise --page-size" >&2
    exit 1
fi
if (( NUM_MESSAGES * ((PAYLOAD_SIZE + 24 + 63) / 64) > 512 )); then
    echo "ERROR: $NUM_MESSAGES messages of $((PAYLOAD_SIZE + 24)) bytes do" \
         "not fit the playback BRAM (512 lines of 64 bytes);" \
         "lower --num-messages to $(( 512 / ((PAYLOAD_SIZE + 24 + 63) / 64) ))" >&2
    exit 1
fi
if [[ "$REQUEST" == "inline" ]] && (( PAYLOAD_SIZE > 8 )); then
    echo "NOTE: --request inline answers only the first 8 payload bytes;" \
         "use --request ring to serve all $PAYLOAD_SIZE"
fi

# ============================================================================
# Helpers (ports from hsb_test_cpu.sh)
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

make_temp_log() {
    mktemp -t "gpu_grid_dispatch_$1_XXXXXX.log"
}

cleanup_temp_logs() {
    local log
    for log in "$EMULATOR_LOG" "$SERVER_LOG"; do
        [[ -n "$log" ]] || continue
        if $SAVE_TEMPS; then
            echo "Kept log: $log"
        else
            rm -f "$log"
        fi
    done
}

# SIGINT first: gpu_grid_dispatch stops its dispatcher, releases the grid and
# prints its summary on SIGINT.  Anything still alive after a grace period is
# killed.
cleanup_pids() {
    local pid
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -INT "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
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

# ============================================================================
# Run
# ============================================================================
do_run() {
    IB_DEVICE=$(detect_ib_device)
    if [[ -z "$BIN_DIR" ]]; then
        BIN_DIR="$CUDA_QUANTUM_DIR/realtime/build/unittests/utils"
    fi
    local emulator_bin="$BIN_DIR/hsb_fpga_emulator"
    local playback_bin="$BIN_DIR/hsb_fpga_playback"
    local example_bin="$EXAMPLE_BIN"
    if [[ -z "$example_bin" ]]; then
        example_bin="$SCRIPT_DIR/build/gpu_grid_dispatch"
        [[ -x "$example_bin" ]] || example_bin="$SCRIPT_DIR/gpu_grid_dispatch"
    fi

    local bins=("$example_bin" "$playback_bin")
    if $EMULATE; then bins+=("$emulator_bin"); fi
    local bin
    for bin in "${bins[@]}"; do
        if [[ ! -x "$bin" ]]; then
            echo "ERROR: $bin not found (see --bin-dir / --example-bin)." >&2
            exit 1
        fi
    done

    PIDS=()
    trap 'cleanup_pids; cleanup_temp_logs' EXIT

    local target_ip target_qp
    if $EMULATE; then
        echo "=== Emulated mode ==="
        echo "--- Starting emulator ---"
        EMULATOR_LOG=$(make_temp_log emulator)
        echo "  Emulator log: $EMULATOR_LOG"
        "$emulator_bin" \
            --device="$IB_DEVICE" \
            --port="$CONTROL_PORT" \
            --bridge-ip="$BRIDGE_IP" \
            --page-size="$PAGE_SIZE" \
            > "$EMULATOR_LOG" 2>&1 &
        PIDS+=($!)
        # Prefix via process substitution rather than a pipeline: $! must be
        # tail's pid so cleanup_pids kills tail and sed sees EOF and flushes.
        tail -f "$EMULATOR_LOG" > >(sed -u 's/^/[EMULATOR] /') &
        PIDS+=($!)

        sleep 2
        target_qp=$(grep -oP 'QP Number: 0x\K[0-9a-fA-F]+' "$EMULATOR_LOG" | head -1 || true)
        if [[ -z "$target_qp" ]]; then
            echo "ERROR: Could not parse emulator QP from log" >&2
            exit 1
        fi
        target_qp="0x$target_qp"
        target_ip="$EMULATOR_IP"
        echo "  Emulator QP: $target_qp"
    else
        echo "=== FPGA mode ==="
        target_qp="$FPGA_QP"
        target_ip="$FPGA_IP"
    fi

    echo "--- Starting gpu_grid_dispatch (notify=$NOTIFY poll=$POLL request=$REQUEST) ---"
    SERVER_LOG=$(make_temp_log server)
    echo "  Server log: $SERVER_LOG"
    local server_args=(
        --gpu="$GPU_ID"
        --notify="$NOTIFY"
        --poll="$POLL"
        --request="$REQUEST"
        --timeout="$TIMEOUT"
    )
    [[ -n "$BLOCKS" ]] && server_args+=(--blocks="$BLOCKS")
    [[ -n "$THREADS" ]] && server_args+=(--threads="$THREADS")
    [[ -n "$BRIDGE_LIB" ]] && server_args+=(--bridge="$BRIDGE_LIB")
    server_args+=(
        --
        --device="$IB_DEVICE"
        --local-ip="$BRIDGE_IP"
        --qp_config=hsb_fpga
        --peer-ip="$target_ip"
        --remote-qp="$target_qp"
        --num-slots="$NUM_PAGES"
        --slot-size="$PAGE_SIZE"
        --unified
    )
    [[ -n "$FRAME_SIZE" ]] && server_args+=(--frame-size="$FRAME_SIZE")
    "$example_bin" "${server_args[@]}" > "$SERVER_LOG" 2>&1 &
    local server_pid=$!
    PIDS+=($server_pid)
    tail -f "$SERVER_LOG" > >(sed -u 's/^/[SERVER] /') &
    PIDS+=($!)

    # The READY line is printed once the grid is resident and before the
    # transport connects; it carries this end's qp / rkey / buffer_addr.
    local wait_elapsed=0
    while ! grep -q "CUDAQ_REALTIME_SERVER_READY" "$SERVER_LOG" 2>/dev/null; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "ERROR: gpu_grid_dispatch died during startup" >&2
            cat "$SERVER_LOG" >&2
            exit 1
        fi
        if (( wait_elapsed >= 30 )); then
            echo "ERROR: gpu_grid_dispatch did not become ready within 30s" >&2
            cat "$SERVER_LOG" >&2
            exit 1
        fi
        sleep 1
        (( wait_elapsed++ )) || true
    done

    local ready qp rkey buffer
    ready=$(grep -m1 "CUDAQ_REALTIME_SERVER_READY" "$SERVER_LOG")
    qp=$(grep -oP '\bqp=0x\K[0-9a-fA-F]+' <<<"$ready" || true)
    rkey=$(grep -oP '\brkey=\K[0-9]+' <<<"$ready" || true)
    buffer=$(grep -oP '\bbuffer_addr=0x\K[0-9a-fA-F]+' <<<"$ready" || true)
    if [[ -z "$qp" || -z "$rkey" || -z "$buffer" ]]; then
        echo "ERROR: Could not parse qp/rkey/buffer_addr from: $ready" >&2
        exit 1
    fi
    echo "  Server QP: 0x$qp  RKey: $rkey  Buffer: 0x$buffer"

    echo "--- Starting playback ---"
    local playback_args=(
        --hsb-ip="$target_ip"
        --bridge-qp="0x$qp"
        --bridge-rkey="$rkey"
        --bridge-buffer="0x$buffer"
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

    local playback_exit=0
    "$playback_bin" "${playback_args[@]}" || playback_exit=$?

    sleep 1
    cleanup_pids   # SIGINT: the server prints requests served and placement

    echo ""
    grep -E "^(handoff|requests served|placement):" "$SERVER_LOG" || true
    if [[ $playback_exit -eq 0 ]]; then
        echo "*** TEST PASSED ***"
    else
        echo "*** TEST FAILED ***"
    fi
    exit $playback_exit
}

echo "=== gpu_grid_dispatch test ==="
echo "Mode: $(if $EMULATE; then echo "emulated"; else echo "FPGA"; fi)"
do_run
