#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Install the RXE (SoftRoCE) userspace libibverbs provider on a DOCA host.
#
# Background
# ----------
# DOCA SDK 3.3.0 replaces Ubuntu's stock ibverbs-providers (rdmav34) with its
# own package (rdmav59) that includes only the mlx5 hardware provider. The RXE
# software provider (librxe-rdmav*.so) is absent, so ibv_get_device_list()
# returns 0 devices even when the kernel rdma_rxe module is loaded and
# rdma link show shows the device as ACTIVE.
#
# Root cause
# ----------
# The provider ABI version (PABI) baked into provider filenames and symbol
# version strings cannot be bridged by renaming:
#
#   Ubuntu ibverbs-providers  → librxe-rdmav34.so imports @IBVERBS_PRIVATE_34
#   DOCA libibverbs.so.1      → exports @IBVERBS_PRIVATE_59  (25-version gap)
#
# Simply renaming librxe-rdmav34.so → librxe-rdmav59.so does NOT work because
# the dynamic linker still looks for @IBVERBS_PRIVATE_34 symbols in libibverbs
# and fails when they are absent.
#
# Strategy
# --------
# Install TWO things side by side:
#
#   1. Ubuntu's libibverbs.so.1 (rdmav34) extracted to /opt/ubuntu-ibverbs/lib/
#      This version exports @IBVERBS_PRIVATE_34 symbols and can load rxe.
#
#   2. Ubuntu's librxe-rdmav34.so installed under its original name alongside
#      DOCA's libmlx5-rdmav59.so in the provider directory.
#      DOCA's libibverbs ignores it (wrong suffix); Ubuntu's libibverbs finds it.
#
# To use: run the daemon / any libibverbs app with:
#   LD_LIBRARY_PATH=/opt/ubuntu-ibverbs/lib:$LD_LIBRARY_PATH <command>
#
# CpuRoceTransceiver only calls the stable public verbs API (ibv_get_device_list,
# ibv_open_device, ibv_alloc_pd, ibv_create_cq, ibv_create_qp, ibv_post_recv,
# ibv_modify_qp, ibv_query_gid_ex, ibv_reg_mr_iova). All of these exist in
# Ubuntu's libibverbs v50.0, so the binary remains ABI-compatible.
#
# Usage
#   sudo bash realtime/scripts/install_rxe_provider.sh

set -euo pipefail

PROVIDER_DIR="/usr/lib/x86_64-linux-gnu/libibverbs"
UBUNTU_IBVERBS_LIB="/opt/ubuntu-ibverbs/lib"
EXTRACT_DIR="/tmp/ibverbs-ubuntu-extract"

# Ubuntu noble-updates packages (same source package: rdma-core 50.0-2ubuntu0.2)
UBUNTU_PROVIDERS_URL="http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/ibverbs-providers_50.0-2ubuntu0.2_amd64.deb"
UBUNTU_LIBIBVERBS_URL="http://archive.ubuntu.com/ubuntu/pool/main/r/rdma-core/libibverbs1_50.0-2ubuntu0.2_amd64.deb"

# ---------------------------------------------------------------------------
# Must run as root
# ---------------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: run as root: sudo bash $0" >&2
  exit 1
fi

echo "=== RXE userspace provider installer (dual-libibverbs strategy) ==="
echo "Provider dir :  ${PROVIDER_DIR}"
echo "Ubuntu ibverbs: ${UBUNTU_IBVERBS_LIB}"

# ---------------------------------------------------------------------------
# Clean up previous (broken) renamed provider if present
# ---------------------------------------------------------------------------
if [[ -f "${PROVIDER_DIR}/librxe-rdmav59.so" ]]; then
  echo "Removing broken librxe-rdmav59.so (ABI-incompatible rename)..."
  rm -f "${PROVIDER_DIR}/librxe-rdmav59.so"
fi

# ---------------------------------------------------------------------------
# Download and extract Ubuntu packages
# ---------------------------------------------------------------------------
rm -rf "${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}/providers" "${EXTRACT_DIR}/libibverbs1"

echo ""
echo "--- Downloading Ubuntu ibverbs-providers ---"
curl -fL "${UBUNTU_PROVIDERS_URL}" -o "${EXTRACT_DIR}/ibverbs-providers.deb"
dpkg-deb -x "${EXTRACT_DIR}/ibverbs-providers.deb" "${EXTRACT_DIR}/providers"

echo "--- Downloading Ubuntu libibverbs1 ---"
curl -fL "${UBUNTU_LIBIBVERBS_URL}" -o "${EXTRACT_DIR}/libibverbs1.deb"
dpkg-deb -x "${EXTRACT_DIR}/libibverbs1.deb" "${EXTRACT_DIR}/libibverbs1"

# ---------------------------------------------------------------------------
# Install Ubuntu's rxe provider under its original (rdmav34) name
# DOCA's libibverbs only loads *-rdmav59.so providers, so it ignores this file.
# Ubuntu's libibverbs (via LD_LIBRARY_PATH) finds it and loads it correctly.
# ---------------------------------------------------------------------------
rxe_so=$(find "${EXTRACT_DIR}/providers" -name "librxe-rdmav*.so" | head -1)
if [[ -z "${rxe_so}" ]]; then
  echo "ERROR: librxe-rdmav*.so not found in Ubuntu ibverbs-providers." >&2
  find "${EXTRACT_DIR}/providers" -name "lib*.so" | head -20 >&2
  exit 1
fi

rxe_basename=$(basename "${rxe_so}")
echo ""
echo "--- Installing ${rxe_basename} (keep original rdmav34 name) ---"
install -Dm755 "${rxe_so}" "${PROVIDER_DIR}/${rxe_basename}"
echo "Installed: ${PROVIDER_DIR}/${rxe_basename}"

# On DOCA/kernel-6.17+ hosts, libibverbs resolves the driver name for rxe
# devices as "mlx5" (not "rxe") — probably because DOCA's kernel patches
# change how the rxe device driver is reported via netlink/sysfs. libibverbs
# then tries to load libmlx5-rdmav34.so. We satisfy that by symlinking the
# rxe provider under that filename. The rxe provider's internal match table
# uses VERBS_NAME_MATCH("rxe", NULL) (prefix match), so it will correctly
# identify rxe_cudaq0 regardless of the .so filename.
echo "--- Creating libmlx5-rdmav34.so → librxe-rdmav34.so symlink (driver name fix) ---"
ln -sf "${PROVIDER_DIR}/${rxe_basename}" "${PROVIDER_DIR}/libmlx5-rdmav34.so"
echo "Symlink: ${PROVIDER_DIR}/libmlx5-rdmav34.so -> ${rxe_basename}"

# ---------------------------------------------------------------------------
# Install Ubuntu's libibverbs.so.1 to /opt/ubuntu-ibverbs/lib/
# This version exports @IBVERBS_PRIVATE_34 symbols that the rxe provider needs.
# ---------------------------------------------------------------------------
ubuntu_libverbs_real=$(find "${EXTRACT_DIR}/libibverbs1" -name "libibverbs.so.1.*" \
                       ! -type l | head -1)
if [[ -z "${ubuntu_libverbs_real}" ]]; then
  # Some deb layouts use a symlink; fall back to any match
  ubuntu_libverbs_real=$(find "${EXTRACT_DIR}/libibverbs1" -name "libibverbs.so.1*" | head -1)
fi
if [[ -z "${ubuntu_libverbs_real}" ]]; then
  echo "ERROR: libibverbs.so.1 not found in Ubuntu libibverbs1 deb." >&2
  find "${EXTRACT_DIR}/libibverbs1" -name "*.so*" | head -20 >&2
  exit 1
fi

echo ""
echo "--- Installing Ubuntu's libibverbs to ${UBUNTU_IBVERBS_LIB}/ ---"
mkdir -p "${UBUNTU_IBVERBS_LIB}"
install -Dm755 "${ubuntu_libverbs_real}" \
  "${UBUNTU_IBVERBS_LIB}/$(basename "${ubuntu_libverbs_real}")"
# Create the soname symlink libibverbs.so.1 -> libibverbs.so.1.x.y
ln -sf "$(basename "${ubuntu_libverbs_real}")" \
  "${UBUNTU_IBVERBS_LIB}/libibverbs.so.1"
echo "Installed: ${UBUNTU_IBVERBS_LIB}/$(basename "${ubuntu_libverbs_real}")"
echo "Symlink:   ${UBUNTU_IBVERBS_LIB}/libibverbs.so.1"

# ---------------------------------------------------------------------------
# Verify: run ibv_devices with Ubuntu's libibverbs
# ---------------------------------------------------------------------------
echo ""
echo "=== Provider directory ==="
ls -la "${PROVIDER_DIR}/"

echo ""
echo "=== Testing with Ubuntu libibverbs (LD_LIBRARY_PATH) ==="
echo "Command: LD_LIBRARY_PATH=${UBUNTU_IBVERBS_LIB} ibv_devices"
if LD_LIBRARY_PATH="${UBUNTU_IBVERBS_LIB}" ibv_devices 2>/dev/null; then
  echo ""
  if LD_LIBRARY_PATH="${UBUNTU_IBVERBS_LIB}" ibv_devices 2>/dev/null | grep -q "rxe"; then
    echo "SUCCESS: rxe device visible with Ubuntu's libibverbs."
  else
    echo "NOTE: ibv_devices ran successfully but no rxe device found."
    echo "Run setup_softroce_loopback.sh first to create the rxe device, then re-check."
  fi
else
  echo "WARNING: ibv_devices returned non-zero with Ubuntu's libibverbs."
fi

echo ""
echo "=== Usage ==="
echo "To run any libibverbs application with SoftRoCE support on this DOCA host:"
echo "  LD_LIBRARY_PATH=${UBUNTU_IBVERBS_LIB} <your-command>"
echo ""
echo "The probe scripts (probe_cpu_roce_gpu_device_call_softroce.sh) set this"
echo "automatically when ${UBUNTU_IBVERBS_LIB} is present."
