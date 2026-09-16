#!/bin/sh

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Retry a command, clearing package-manager metadata between attempts. The CUDA
# yum repo CDN intermittently serves a stale repomd.xml that points at rotated
# repodata files, producing 404s; clearing metadata forces a fresh fetch.
function retry {
  local n=0 max=5 delay=15
  until "$@"; do
    n=$((n+1))
    if [ "$n" -ge "$max" ]; then
      echo "Command failed after $max attempts: $*" >&2
      return 1
    fi
    echo "Attempt $n/$max failed; clearing repo metadata and retrying in ${delay}s..." >&2
    if [ -x "$(command -v dnf)" ]; then dnf clean all || true
    elif [ -x "$(command -v apt-get)" ]; then apt-get clean || true; fi
    sleep "$delay"
  done
}


# Version pins and helpers shared by the CUDA-Q Realtime dependency scripts,
# i.e., install_dev_prerequisites.sh (standard apt path) and install_devdeps.sh
# (containers that already ship Mellanox OFED).
#
# Usage:
# This file is meant to be sourced, not executed:
#   . "$(dirname "$0")/deps_common.sh"

# Central repository of DOCA and Holoscan SDK versions for development and CI.
DOCA_VERSION=${DOCA_VERSION:-3.3.0}
HOLOSCAN_SDK_VERSION=${HOLOSCAN_SDK_VERSION:-4.6.0.0}
HOLOSCAN_SDK_INSTALL_PREFIX=${HOLOSCAN_SDK_INSTALL_PREFIX:-/opt/nvidia/holoscan}
CUDAQ_REALTIME_HSB_REPO=${CUDAQ_REALTIME_HSB_REPO:-https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git}
CUDAQ_REALTIME_HSB_REF=${CUDAQ_REALTIME_HSB_REF:-2.6.0-EA2}

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

# Fetch a URL to a file with whichever downloader the image ships: the CI
# containers have curl, the RHEL assets image has wget.
cudaq_realtime_download() {
  if [ -x "$(command -v curl)" ]; then
    curl -fsSL "$1" -o "$2"
  elif [ -x "$(command -v wget)" ]; then
    wget -q "$1" -O "$2"
  else
    echo "Neither curl nor wget is available to download $1" >&2
    return 1
  fi
}

# Directory naming the DOCA repository uses for this distro. Ubuntu is named by
# id and version, while the RHEL family is served by major version alone under
# the id of the distro it is built for rather than the one running.
cudaq_realtime_doca_distro() {
  _cudaq_realtime_doca_id=$(. /etc/os-release && printf '%s' "$ID")
  _cudaq_realtime_doca_version_id=$(. /etc/os-release && printf '%s' "$VERSION_ID")
  case "$_cudaq_realtime_doca_id" in
    ubuntu) printf 'ubuntu%s' "$_cudaq_realtime_doca_version_id" ;; # e.g., ubuntu24.04
    rhel | almalinux | rocky | centos)
      printf 'rhel%s' "${_cudaq_realtime_doca_version_id%%.*}" ;;   # e.g., rhel8
    *)
      echo "No DOCA repository is published for $_cudaq_realtime_doca_id$_cudaq_realtime_doca_version_id" >&2
      return 1
      ;;
  esac
}

# Register the DOCA host package repository for this architecture and distro.
# The same repository serves apt and dnf, so both entry points and the RHEL
# assets image share one source: on RHEL this replaces fetching the ~640M
# doca-host package, which is itself only an offline copy of this repository.
cudaq_realtime_add_doca_repo() {
  echo "Installing DOCA version $DOCA_VERSION..."
  _cudaq_realtime_doca_arch=$(uname -m)
  case "$_cudaq_realtime_doca_arch" in
    aarch64 | arm64) _cudaq_realtime_doca_arch="arm64-sbsa" ;;
  esac
  _cudaq_realtime_distro=$(cudaq_realtime_doca_distro) || return 1
  export DOCA_URL="https://linux.mellanox.com/public/repo/doca/$DOCA_VERSION/$_cudaq_realtime_distro/$_cudaq_realtime_doca_arch/"
  echo "Using DOCA_REPO_LINK=${DOCA_URL}"

  if [ -x "$(command -v apt-get)" ]; then
    if [ ! -x "$(command -v curl)" ] || [ ! -x "$(command -v gpg)" ]; then
      retry apt-get update
      retry apt-get install -y --no-install-recommends curl gnupg
    fi
    curl https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" > /etc/apt/sources.list.d/doca.list
    retry apt-get update
  elif [ -x "$(command -v dnf)" ]; then
    # dnf fetches the key itself, so nothing has to be installed to set this up.
    # Taken from the repository directory rather than the root, because this is
    # where the key that signs these packages lives. Note that DOCA 3.4 renamed
    # it to doca_keyring.gpg, so raising the pin means revisiting this line.
    cat > /etc/yum.repos.d/doca.repo <<EOF
[doca]
name=NVIDIA DOCA $DOCA_VERSION
baseurl=$DOCA_URL
enabled=1
gpgcheck=1
gpgkey=${DOCA_URL}GPG-KEY-Mellanox.pub
EOF
    retry dnf -y makecache
  else
    echo "No supported package manager to register the DOCA repository with." >&2
    return 1
  fi
}

# Install the Holoscan SDK matching the CUDA toolkit in use, from the redist
# archive rather than from apt, the way realtime/docker/assets.Dockerfile
# already installs it on RHEL. The archive carries its own dependencies, so it
# neither pulls in the Ubuntu packages that conflict with a container's
# Mellanox OFED -- which is what the apt path needed a dependency-forced dpkg
# install to work around -- nor ties the version to what a distro repository
# happens to be serving.
cudaq_realtime_install_holoscan() {
  _cudaq_realtime_holoscan_cuda_major=$(cudaq_realtime_cuda_major) || return 1
  _cudaq_realtime_holoscan_arch=$(uname -m)
  case "$_cudaq_realtime_holoscan_arch" in
    aarch64 | arm64) _cudaq_realtime_holoscan_arch=sbsa ;;
    *) _cudaq_realtime_holoscan_arch=x86_64 ;;
  esac
  _cudaq_realtime_holoscan_archive="holoscan-linux-$_cudaq_realtime_holoscan_arch-${HOLOSCAN_SDK_VERSION}_cuda$_cudaq_realtime_holoscan_cuda_major-archive.tar.xz"
  _cudaq_realtime_holoscan_url="https://developer.download.nvidia.com/compute/holoscan/redist/holoscan/linux-$_cudaq_realtime_holoscan_arch/$_cudaq_realtime_holoscan_archive"

  echo "Installing Holoscan SDK $HOLOSCAN_SDK_VERSION from $_cudaq_realtime_holoscan_url"
  _cudaq_realtime_holoscan_tmp=$(mktemp -d)
  # Downloaded whole before it is unpacked, so a truncated transfer costs a
  # retry rather than leaving a half-populated prefix behind.
  retry cudaq_realtime_download "$_cudaq_realtime_holoscan_url" \
    "$_cudaq_realtime_holoscan_tmp/holoscan.tar.xz" &&
    mkdir -p "$HOLOSCAN_SDK_INSTALL_PREFIX" &&
    tar xf "$_cudaq_realtime_holoscan_tmp/holoscan.tar.xz" \
      --strip-components 1 -C "$HOLOSCAN_SDK_INSTALL_PREFIX"
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
  if [ ! -d "$HOLOSCAN_SDK_INSTALL_PREFIX/include" ]; then
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

  # The CUDA-free HololinkRoce leaf exports its package during configure, but
  # no target below depends on it, so name it explicitly when the ref has it.
  local hololink_roce_targets=()
  if [ -d "$HSB_ROOT/src/hololink/transport/roce" ]; then
    hololink_roce_targets=(hololink_transport_roce)
  fi

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
    --target roce_receiver gpu_roce_transceiver hololink_core \
    "${hololink_roce_targets[@]}"
  echo "holoscan-sensor-bridge built at $HSB_BUILD"
}
