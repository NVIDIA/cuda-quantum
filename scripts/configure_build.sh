#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

trap '(return 0 2>/dev/null) && return 1 || exit 1' ERR

# [>InstallLocations]
export CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
export CUQUANTUM_INSTALL_PREFIX=/usr/local/cuquantum
export CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor
export LLVM_INSTALL_PREFIX=/usr/local/llvm
export BLAS_INSTALL_PREFIX=/usr/local/blas
export ZLIB_INSTALL_PREFIX=/usr/local/zlib
export OPENSSL_INSTALL_PREFIX=/usr/local/openssl
export CURL_INSTALL_PREFIX=/usr/local/curl
export AWS_INSTALL_PREFIX=/usr/local/aws

# [<InstallLocations]

if [ "$1" == "install-cuda" ]; then
    DISTRIBUTION=${DISTRIBUTION:-rhel8}
    CUDA_ARCH_FOLDER=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64)

# [>CUDAInstall]
    CUDA_VERSION=${CUDA_VERSION:-12.6}
    CUDA_DOWNLOAD_URL=https://developer.download.nvidia.com/compute/cuda/repos
    # Go to the url above, set the variables below to a suitable distribution
    # and subfolder for your platform, and uncomment the line below.
    # DISTRIBUTION=rhel8 CUDA_ARCH_FOLDER=x86_64

    dnf config-manager --add-repo "${CUDA_DOWNLOAD_URL}/${DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${DISTRIBUTION}.repo"
    dnf install -y --nobest --setopt=install_weak_deps=False \
        cuda-toolkit-$(echo ${CUDA_VERSION} | tr . -)
    # custatevec is now linked to `libnvidia-ml.so.1`, which is provided in the NVIDIA driver.
    # For build on non-GPU systems, we also need to install the driver. 
    dnf install -y --nobest --setopt=install_weak_deps=False nvidia-driver-libs    
# [<CUDAInstall]
fi

if [ "$1" == "install-cudart" ]; then
    DISTRIBUTION=${DISTRIBUTION:-rhel8}
    CUDA_ARCH_FOLDER=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64)

# [>CUDARTInstall]
    CUDA_VERSION=${CUDA_VERSION:-12.6}
    CUDA_DOWNLOAD_URL=https://developer.download.nvidia.com/compute/cuda/repos
    # Go to the url above, set the variables below to a suitable distribution
    # and subfolder for your platform, and uncomment the line below.
    # DISTRIBUTION=rhel8 CUDA_ARCH_FOLDER=x86_64

    version_suffix=$(echo ${CUDA_VERSION} | tr . -)
    dnf config-manager --add-repo "${CUDA_DOWNLOAD_URL}/${DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${DISTRIBUTION}.repo"
    dnf install -y --nobest --setopt=install_weak_deps=False \
        cuda-cudart-${version_suffix} \
        cuda-nvrtc-${version_suffix} \
        libcusolver-${version_suffix} \
        libcusparse-${version_suffix} \
        libcublas-${version_suffix} \
        libcurand-${version_suffix}
    if [ $(echo ${CUDA_VERSION} | cut -d . -f1) -gt 11 ]; then 
        dnf install -y --nobest --setopt=install_weak_deps=False \
            libnvjitlink-${version_suffix}
    fi
# [<CUDARTInstall]
fi

if [ "$1" == "install-cuquantum" ]; then
    CUDA_VERSION=${CUDA_VERSION:-12.6}
    CUDA_ARCH_FOLDER=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64)

# [>cuQuantumInstall]
    CUQUANTUM_VERSION=25.09.1.12
    CUQUANTUM_DOWNLOAD_URL=https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum

    cuquantum_archive=cuquantum-linux-${CUDA_ARCH_FOLDER}-${CUQUANTUM_VERSION}_cuda$(echo ${CUDA_VERSION} | cut -d . -f1)-archive.tar.xz
    wget "${CUQUANTUM_DOWNLOAD_URL}/linux-${CUDA_ARCH_FOLDER}/${cuquantum_archive}"
    mkdir -p "${CUQUANTUM_INSTALL_PREFIX}" 
    tar xf "${cuquantum_archive}" --strip-components 1 -C "${CUQUANTUM_INSTALL_PREFIX}" 
    rm -rf "${cuquantum_archive}"
# [<cuQuantumInstall]
fi

if [ "$1" == "install-cutensor" ]; then
    CUDA_VERSION=${CUDA_VERSION:-12.6}
    CUDA_ARCH_FOLDER=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64)

# [>cuTensorInstall]
    CUTENSOR_VERSION=2.3.1.0
    CUTENSOR_DOWNLOAD_URL=https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor

    cutensor_archive=libcutensor-linux-${CUDA_ARCH_FOLDER}-${CUTENSOR_VERSION}_cuda$(echo ${CUDA_VERSION} | cut -d . -f1)-archive.tar.xz
    wget "${CUTENSOR_DOWNLOAD_URL}/linux-${CUDA_ARCH_FOLDER}/${cutensor_archive}"
    mkdir -p "${CUTENSOR_INSTALL_PREFIX}" && tar xf "${cutensor_archive}" --strip-components 1 -C "${CUTENSOR_INSTALL_PREFIX}"
    rm -rf "${cutensor_archive}"
# [<cuTensorInstall]
fi

if [ "$1" == "install-gcc" ]; then
# [>gccInstall]
    GCC_VERSION=${GCC_VERSION:-11}
    dnf install -y --nobest --setopt=install_weak_deps=False \
        gcc-toolset-${GCC_VERSION}
    # Enabling the toolchain globally is only needed for debug builds
    # to ensure that the correct assembler is picked to process debug symbols.
    enable_script=`find / -path '*gcc*' -path '*'$GCC_VERSIONS'*' -name enable`
    if [ -n "$enable_script" ]; then
        . "$enable_script"
    fi
# [<gccInstall]
fi

# [>ToolchainConfiguration]
export GCC_TOOLCHAIN=/opt/rh/gcc-toolset-11/root/usr/
export CXX="${GCC_TOOLCHAIN}/bin/g++"
export CC="${GCC_TOOLCHAIN}/bin/gcc"
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDAHOSTCXX="${GCC_TOOLCHAIN}/bin/g++"
# [<ToolchainConfiguration]

if [ "$1" == "build-openmpi" ]; then
    source "${GCC_TOOLCHAIN}/../../enable"

# [>OpenMPIBuild]
    OPENMPI_VERSION=4.1.4
    OPENMPI_DOWNLOAD_URL=https://github.com/open-mpi/ompi

    wget "${OPENMPI_DOWNLOAD_URL}/archive/v${OPENMPI_VERSION}.tar.gz" -O /tmp/openmpi.tar.gz
    mkdir -p ~/.openmpi-src && tar xf /tmp/openmpi.tar.gz --strip-components 1 -C ~/.openmpi-src
    rm -rf /tmp/openmpi.tar.gz && cd ~/.openmpi-src
    ./autogen.pl 
    LDFLAGS=-Wl,--as-needed ./configure \
        --prefix=/usr/local/openmpi \
        --disable-getpwuid --disable-static \
        --disable-debug --disable-mem-debug --disable-event-debug \
        --disable-mem-profile --disable-memchecker \
        --without-verbs \
        --with-cuda=/usr/local/cuda
    make -j$(nproc) 
    make -j$(nproc) install
    cd - && rm -rf ~/.openmpi-src
# [<OpenMPIBuild]
fi

trap - ERR
(return 0 2>/dev/null) && return 0 || exit 0