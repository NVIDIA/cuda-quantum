#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Probe Chuck's desktop fallback path: run a GPU-targeted CUDA-Q device_call app
# while routing the device_call transport through CpuRoceChannel over SoftRoCE.
# This does not use DOCA/GpuRoceTransceiver. It checks whether GPU-side tests can
# funnel through the CPU RoCE transport on a single developer machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEVICE="rxe_cudaq0"
IP="10.88.0.1"
GPU_ID="0"
APP_TARGET="nvidia"
BUILD_DIR="${REPO_ROOT}/build"
SETUP_SOFTROCE=true
CLEANUP_SOFTROCE=true
VERIFY_PINGPONG=true
DO_BUILD=0
LOG_FILE="/tmp/cudaq_cpu_roce_gpu_device_call_softroce.log"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --build-dir DIR       CUDA-Q build directory containing bin/nvq++
                        (default: ${BUILD_DIR})
  --device NAME         SoftRoCE IB device (default: ${DEVICE})
  --ip ADDR             SoftRoCE IPv4 address (default: ${IP})
  --gpu N               CUDA GPU id exposed via CUDA_VISIBLE_DEVICES (default: ${GPU_ID})
  --app-target TARGET   nvq++ target for the caller app (default: ${APP_TARGET})
  --build               Ask cpu_roce_device_call_test.sh to build needed test targets
  --log-file PATH       Probe log path (default: ${LOG_FILE})
  --no-setup            Do not create rxe_cudaq0 first
  --no-cleanup          Do not clean up SoftRoCE after the probe
  --no-verify-pingpong  Skip setup_softroce_loopback.sh --verify-pingpong
  --help, -h            Show this help

This is the SoftRoCE + CpuRoceTransceiver path for GPU-targeted desktop tests.
A passing run proves the caller app can target ${APP_TARGET} while device_call
traffic goes over cpu_roce/rxe_cudaq0 to the separate daemon process.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift ;;
    --device) DEVICE="$2"; shift ;;
    --ip) IP="$2"; shift ;;
    --gpu) GPU_ID="$2"; shift ;;
    --app-target) APP_TARGET="$2"; shift ;;
    --build) DO_BUILD=1 ;;
    --log-file) LOG_FILE="$2"; shift ;;
    --no-setup) SETUP_SOFTROCE=false ;;
    --no-cleanup) CLEANUP_SOFTROCE=false ;;
    --no-verify-pingpong) VERIFY_PINGPONG=false ;;
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

TEST_SCRIPT="${REPO_ROOT}/unittests/device_call/cpu_roce_device_call_test.sh"
NVQPP="${BUILD_DIR}/bin/nvq++"
DAEMON="${BUILD_DIR}/unittests/cpu_roce_test_daemon"

if [[ ! -x "${TEST_SCRIPT}" ]]; then
  echo "ERROR: test script not found or not executable: ${TEST_SCRIPT}" >&2
  exit 2
fi
if [[ ! -x "${NVQPP}" ]]; then
  echo "ERROR: nvq++ not found: ${NVQPP}" >&2
  echo "Build CUDA-Q first or pass --build-dir." >&2
  exit 2
fi
if [[ ! -x "${DAEMON}" && "${DO_BUILD}" != "1" ]]; then
  echo "ERROR: daemon not found: ${DAEMON}" >&2
  echo "Run with --build or build target cpu_roce_test_daemon first." >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
else
  echo "WARNING: nvidia-smi is not available; continuing anyway." >&2
fi

if $SETUP_SOFTROCE; then
  setup_args=(--cleanup-first --self-loop --no-exports)
  if $VERIFY_PINGPONG; then
    setup_args+=(--verify-pingpong)
  fi
  sudo bash "${SCRIPT_DIR}/setup_softroce_loopback.sh" "${setup_args[@]}"
fi

if ! ibv_devices | awk 'NR > 2 { print $1 }' | grep -qx "${DEVICE}"; then
  echo "ERROR: ${DEVICE} is not visible through ibv_devices." >&2
  echo "This usually means the host libibverbs provider stack cannot see RXE." >&2
  ibv_devices >&2 || true
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
rm -f "${LOG_FILE}"

echo "=== CPU RoCE GPU-targeted device_call over SoftRoCE probe ==="
echo "device=${DEVICE} ip=${IP} gpu=${GPU_ID} app_target=${APP_TARGET}"
echo "build_dir=${BUILD_DIR}"
echo "log_file=${LOG_FILE}"

cmd=("${TEST_SCRIPT}"
  --build-dir "${BUILD_DIR}"
  --channel-device "${DEVICE}"
  --channel-ip "${IP}"
  --daemon-device "${DEVICE}"
  --daemon-ip "${IP}"
  --app
  --app-target "${APP_TARGET}")
if [[ "${DO_BUILD}" == "1" ]]; then
  cmd+=(--build)
fi

set +e
"${cmd[@]}" >"${LOG_FILE}" 2>&1
rc=$?
set -e

cat "${LOG_FILE}" || true

if [[ ${rc} -eq 0 ]] && grep -q "=== PASS: device_call over cpu_roce returned 42" "${LOG_FILE}"; then
  echo "CPU_ROCE_GPU_DEVICE_CALL_SOFTROCE_RESULT=pass"
  echo "GPU-targeted device_call app ran through cpu_roce on ${DEVICE}."
  exit 0
fi

echo "CPU_ROCE_GPU_DEVICE_CALL_SOFTROCE_RESULT=fail rc=${rc}"
echo "GPU-targeted device_call app did not pass through cpu_roce on ${DEVICE}."
exit 1
