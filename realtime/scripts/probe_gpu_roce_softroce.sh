#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Probe whether the Hololink/DOCA GpuRoceTransceiver can start on a SoftRoCE
# RXE device. This is intentionally a narrow compatibility check: reaching
# "Bridge Ready" means the GPU/DOCA bridge accepted rxe_cudaq0 far enough to
# expose QP/rkey/buffer information. Failure before that is useful evidence that
# SoftRoCE is only viable for the CPU/libibverbs path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEVICE="rxe_cudaq0"
IP="10.88.0.1"
GPU_ID="0"
REMOTE_QP="0x2"
TIMEOUT_SEC="5"
PAGE_SIZE="384"
NUM_PAGES="64"
BRIDGE_BIN="${REPO_ROOT}/realtime/build/unittests/utils/hololink_bridge"
SETUP_SOFTROCE=true
CLEANUP_SOFTROCE=true
LOG_FILE="/tmp/cudaq_gpu_roce_softroce_probe.log"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --bridge-bin PATH   hololink_bridge binary to run
                      (default: ${BRIDGE_BIN})
  --device NAME       SoftRoCE IB device (default: ${DEVICE})
  --ip ADDR           SoftRoCE IPv4 address / peer IP (default: ${IP})
  --gpu N             CUDA GPU id for DOCA GPUNetIO (default: ${GPU_ID})
  --remote-qp N       Remote QP passed to GpuRoceTransceiver (default: ${REMOTE_QP})
  --timeout N         Bridge runtime timeout in seconds (default: ${TIMEOUT_SEC})
  --page-size N       Ring slot size passed to hololink_bridge (default: ${PAGE_SIZE})
  --num-pages N       Ring slot count passed to hololink_bridge (default: ${NUM_PAGES})
  --log-file PATH     Probe log path (default: ${LOG_FILE})
  --no-setup          Do not create rxe_cudaq0 first
  --no-cleanup        Do not clean up SoftRoCE after the probe
  --help, -h          Show this help

The expected setup is a GPU/DOCA/HSB-capable host with hololink_bridge already
built. The script creates a single-device SoftRoCE self-loop by default, then
runs hololink_bridge in forward mode against that RXE device.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bridge-bin) BRIDGE_BIN="$2"; shift ;;
    --device) DEVICE="$2"; shift ;;
    --ip) IP="$2"; shift ;;
    --gpu) GPU_ID="$2"; shift ;;
    --remote-qp) REMOTE_QP="$2"; shift ;;
    --timeout) TIMEOUT_SEC="$2"; shift ;;
    --page-size) PAGE_SIZE="$2"; shift ;;
    --num-pages) NUM_PAGES="$2"; shift ;;
    --log-file) LOG_FILE="$2"; shift ;;
    --no-setup) SETUP_SOFTROCE=false ;;
    --no-cleanup) CLEANUP_SOFTROCE=false ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cleanup() {
  if $CLEANUP_SOFTROCE; then
    sudo bash "${SCRIPT_DIR}/setup_softroce_loopback.sh" --cleanup --use-netns >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ ! -x "${BRIDGE_BIN}" ]]; then
  cat >&2 <<EOF
ERROR: hololink_bridge binary not found or not executable:
  ${BRIDGE_BIN}

Build realtime with Hololink tools enabled first, for example:
  bash realtime/unittests/utils/hololink_test.sh --build --no-run \
    --cuda-quantum-dir "${REPO_ROOT}" \
    --hololink-dir /path/to/holoscan-sensor-bridge
EOF
  exit 2
fi

if ! command -v ibv_devices >/dev/null 2>&1; then
  echo "WARNING: ibv_devices is not available; will use 'rdma link' for device check." >&2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi is not available; continuing, but this is not a GPU-capable environment." >&2
else
  nvidia-smi -L || true
fi

if $SETUP_SOFTROCE; then
  # Skip --verify-pingpong: DOCA's libibverbs drops RXE (software RoCE) provider
  # support, so ibv_uc_pingpong cannot find the RXE device. The kernel rdma link
  # state is used for verification instead.
  sudo bash "${SCRIPT_DIR}/setup_softroce_loopback.sh" \
    --cleanup-first \
    --self-loop \
    --no-exports
fi

# Use 'rdma link' instead of ibv_devices: DOCA's libibverbs does not include the
# RXE userspace provider, so ibv_devices cannot enumerate software RoCE devices.
if ! rdma link show 2>/dev/null | awk '{print $2}' | cut -d/ -f1 | grep -qx "${DEVICE}"; then
  echo "ERROR: ${DEVICE} is not visible through 'rdma link show'." >&2
  rdma link show >&2 || true
  exit 2
fi

rm -f "${LOG_FILE}"
echo "=== GpuRoceTransceiver SoftRoCE probe ==="
echo "device=${DEVICE} ip=${IP} gpu=${GPU_ID} remote_qp=${REMOTE_QP} timeout=${TIMEOUT_SEC}s"
echo "bridge_bin=${BRIDGE_BIN}"
echo "log_file=${LOG_FILE}"

set +e
CUDA_MODULE_LOADING=EAGER timeout "$((TIMEOUT_SEC + 30))s" "${BRIDGE_BIN}" \
  --device="${DEVICE}" \
  --peer-ip="${IP}" \
  --remote-qp="${REMOTE_QP}" \
  --gpu="${GPU_ID}" \
  --timeout="${TIMEOUT_SEC}" \
  --page-size="${PAGE_SIZE}" \
  --num-pages="${NUM_PAGES}" \
  --forward \
  >"${LOG_FILE}" 2>&1
bridge_rc=$?
set -e

cat "${LOG_FILE}" || true

if grep -q "=== Bridge Ready ===" "${LOG_FILE}"; then
  echo "GPU_ROCE_SOFTROCE_RESULT=bridge-ready"
  echo "GpuRoceTransceiver reached Bridge Ready on ${DEVICE}."
  exit 0
fi

echo "GPU_ROCE_SOFTROCE_RESULT=bridge-not-ready rc=${bridge_rc}"
echo "GpuRoceTransceiver did not reach Bridge Ready on ${DEVICE}."
exit 1
