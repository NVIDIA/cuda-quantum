#!/bin/sh

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Version pins and helpers shared by the CUDA-Q Realtime dependency scripts,
# i.e., install_dev_prerequisites.sh (standard apt path) and install_devdeps.sh
# (containers that already ship Mellanox OFED).
#
# Usage:
# This file is meant to be sourced, not executed:
#   . "$(dirname "$0")/deps_common.sh"

CUDAQ_REALTIME_DOCA_VERSION=3.3.0
CUDAQ_REALTIME_HSB_REPO=https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
CUDAQ_REALTIME_HSB_REF=2.6.0-EA2

# Major CUDA version reported by nvcc, e.g., 13.
cudaq_realtime_cuda_major() {
  _cudaq_realtime_cuda_major=$(nvcc --version 2>/dev/null |
    sed -n 's/^.*release \([0-9]\+\).*$/\1/p')
  if [ -z "$_cudaq_realtime_cuda_major" ]; then
    echo "Could not determine CUDA version from nvcc. Is the CUDA toolkit installed?" >&2
    echo "CUDA-Q Realtime requires CUDA toolkit to be installed." >&2
    return 1
  fi
  printf '%s' "$_cudaq_realtime_cuda_major"
}

# Full CUDA version reported by nvcc, e.g., 13.0.
cudaq_realtime_cuda_version() {
  _cudaq_realtime_cuda_version=$(nvcc --version 2>/dev/null |
    sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
  if [ -z "$_cudaq_realtime_cuda_version" ]; then
    echo "Could not determine CUDA version from nvcc. Is the CUDA toolkit installed?" >&2
    echo "CUDA-Q Realtime requires CUDA toolkit to be installed." >&2
    return 1
  fi
  printf '%s' "$_cudaq_realtime_cuda_version"
}

# CUDA architectures HSB is compiled for. HSB reads CUDA_NATIVE_ARCH from the
# environment, so a value set by the caller always wins.
cudaq_realtime_cuda_native_arch() {
  if [ -n "${CUDA_NATIVE_ARCH:-}" ]; then
    printf '%s' "$CUDA_NATIVE_ARCH"
    return 0
  fi
  _cudaq_realtime_arch_cuda_major=$(cudaq_realtime_cuda_major) || return 1
  if [ "$_cudaq_realtime_arch_cuda_major" = 12 ]; then
    printf '%s' "80-real;90"
  else
    printf '%s' "80-real;90-real;100f-real;110-real;120-real;100-virtual"
  fi
}

# Register the DOCA host apt repository for this architecture and distro.
cudaq_realtime_add_doca_repo() {
  if [ ! -x "$(command -v curl)" ] || [ ! -x "$(command -v gpg)" ]; then
    apt-get update && apt-get install -y --no-install-recommends curl gnupg
  fi

  echo "Installing DOCA version $CUDAQ_REALTIME_DOCA_VERSION..."
  _cudaq_realtime_doca_arch=$(uname -m)
  case "$_cudaq_realtime_doca_arch" in
    aarch64 | arm64) _cudaq_realtime_doca_arch="arm64-sbsa" ;;
  esac
  _cudaq_realtime_distro=$(. /etc/os-release && echo ${ID}${VERSION_ID}) # e.g., ubuntu24.04
  export DOCA_URL="https://linux.mellanox.com/public/repo/doca/$CUDAQ_REALTIME_DOCA_VERSION/$_cudaq_realtime_distro/$_cudaq_realtime_doca_arch/"
  echo "Using DOCA_REPO_LINK=${DOCA_URL}"
  curl https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub
  echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" > /etc/apt/sources.list.d/doca.list
  apt-get update
}

# Install the Holoscan SDK matching the CUDA toolkit in use. Set
# CUDAQ_REALTIME_HOLOSCAN_FORCE_DEPS=1 to fall back to a dependency-forced dpkg
# install; needed in containers whose pre-installed packages keep apt from
# resolving the Holoscan dependency chain.
cudaq_realtime_install_holoscan() {
  _cudaq_realtime_holoscan_cuda_major=$(cudaq_realtime_cuda_major) || return 1
  apt-get update
  if apt-get install -y --no-install-recommends \
    holoscan-cuda-$_cudaq_realtime_holoscan_cuda_major; then
    return 0
  fi
  if [ "${CUDAQ_REALTIME_HOLOSCAN_FORCE_DEPS:-0}" != 1 ]; then
    return 1
  fi
  _cudaq_realtime_holoscan_tmp=$(mktemp -d)
  (cd "$_cudaq_realtime_holoscan_tmp" &&
    apt-get download holoscan holoscan-cuda-$_cudaq_realtime_holoscan_cuda_major &&
    dpkg --force-depends -i holoscan*.deb)
  _cudaq_realtime_holoscan_status=$?
  rm -rf "$_cudaq_realtime_holoscan_tmp"
  return $_cudaq_realtime_holoscan_status
}

# Fail early if DOCA or the Holoscan SDK did not land where HSB expects them.
cudaq_realtime_verify_sdks() {
  if [ ! -d /opt/mellanox/doca/include ]; then
    echo "ERROR: DOCA SDK installation failed" >&2
    return 1
  fi
  if [ ! -d /opt/nvidia/holoscan ]; then
    echo "ERROR: Holoscan SDK installation failed" >&2
    return 1
  fi
}

# Clone and build the Holoscan Sensor Bridge libraries CUDA-Q Realtime links
# against. HSB_ROOT selects the source tree (default /tmp/holoscan-sensor-bridge)
# and the build lands in $HSB_ROOT/build; callers pass both to CMake through
# HOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR and HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR.
# Set CUDAQ_REALTIME_HSB_STRIP_OPERATORS=1 to drop the operators CUDA-Q Realtime
# does not use.
cudaq_realtime_build_hsb() {
  export HSB_ROOT="${HSB_ROOT:-/tmp/holoscan-sensor-bridge}"
  export HSB_BUILD="${HSB_ROOT}/build"

  CUDA_NATIVE_ARCH=$(cudaq_realtime_cuda_native_arch) || return 1
  export CUDA_NATIVE_ARCH
  echo "Building holoscan-sensor-bridge $CUDAQ_REALTIME_HSB_REF for CUDA_NATIVE_ARCH=$CUDA_NATIVE_ARCH"

  rm -rf "$HSB_ROOT"
  git clone --depth 1 --branch "$CUDAQ_REALTIME_HSB_REF" \
    "$CUDAQ_REALTIME_HSB_REPO" "$HSB_ROOT"

  if [ "${CUDAQ_REALTIME_HSB_STRIP_OPERATORS:-0}" = 1 ]; then
    # Strip operators we don't need to avoid configure failures from missing deps
    sed -i '/add_subdirectory(audio_packetizer)/d; /add_subdirectory(compute_crc)/d;
            /add_subdirectory(csi_to_bayer)/d; /add_subdirectory(image_processor)/d;
            /add_subdirectory(iq_dec)/d; /add_subdirectory(iq_enc)/d;
            /add_subdirectory(linux_coe_receiver)/d; /add_subdirectory(linux_receiver)/d;
            /add_subdirectory(packed_format_converter)/d; /add_subdirectory(sub_frame_combiner)/d;
            /add_subdirectory(udp_transmitter)/d; /add_subdirectory(emulator)/d;
            /add_subdirectory(sig_gen)/d; /add_subdirectory(sig_viewer)/d' \
      "$HSB_ROOT/src/hololink/operators/CMakeLists.txt"
  fi

  cmake -G Ninja -S "$HSB_ROOT" -B "$HSB_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DHOLOLINK_BUILD_ONLY_NATIVE=OFF \
    -DHOLOLINK_BUILD_PYTHON=OFF \
    -DHOLOLINK_BUILD_TESTS=OFF \
    -DHOLOLINK_BUILD_TOOLS=OFF \
    -DHOLOLINK_BUILD_EXAMPLES=OFF \
    -DHOLOLINK_BUILD_EMULATOR=OFF
  cmake --build "$HSB_BUILD" \
    --target roce_receiver gpu_roce_transceiver hololink_core
  echo "holoscan-sensor-bridge built at $HSB_BUILD"
}
