#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [>InstallLocations]
export CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
export CUQUANTUM_INSTALL_PREFIX=/usr/local/cuquantum
export CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor
export LLVM_INSTALL_PREFIX=/usr/local/llvm
export OPENSSL_INSTALL_PREFIX=/usr/local/openssl
export BLAS_INSTALL_PREFIX=/usr/local/blas
# [<InstallLocations]

if [ "$1" == "install-cuda" ]; then
# [>CUDAInstall]
    CUDA_VERSION=11.8
    CUDA_DOWNLOAD_URL=https://developer.download.nvidia.com/compute/cuda
    # Go to the url above and set the variables below 
    # to the distribution and subfolder for your platform.
    DISTRIBUTION=rhel8 CUDA_ARCH_FOLDER=x86_64

    dnf config-manager --add-repo "${CUDA_DOWNLOAD_URL}/repos/${DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${DISTRIBUTION}.repo"
    dnf install -y --nobest --setopt=install_weak_deps=False \
        cuda-toolkit-$(echo ${CUDA_VERSION} | tr . -)
# [<CUDAInstall]
fi

if [ "$1" == "install-cudart" ]; then
# [>CUDARTInstall]
    CUDA_VERSION=11.8
    CUDA_DOWNLOAD_URL=https://developer.download.nvidia.com/compute/cuda
    # Go to the url above and set the variables below 
    # to the distribution and subfolder for your platform.
    DISTRIBUTION=rhel8 CUDA_ARCH_FOLDER=x86_64

    version_suffix=$(echo ${CUDA_VERSION} | tr . -)
    dnf config-manager --add-repo "${CUDA_DOWNLOAD_URL}/repos/${DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${DISTRIBUTION}.repo"
    dnf install -y --nobest --setopt=install_weak_deps=False \
        cuda-nvtx-${version_suffix} libcusolver-${version_suffix}
# [<CUDARTInstall]
        # libcublas-dev-${version_suffix} 
fi

if [ "$1" == "install-cuquantum" ]; then
    CUDA_VERSION=11.8 CUDA_ARCH_FOLDER=x86_64

# [>cuQuantumInstall]
    CUQUANTUM_VERSION=23.10.0.6
    CUQUANTUM_DOWNLOAD_URL=https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum

    cuquantum_archive=cuquantum-linux-${CUDA_ARCH_FOLDER}-${CUQUANTUM_VERSION}_cuda$(echo ${CUDA_VERSION} | cut -d . -f1)-archive.tar.xz
    wget "${CUQUANTUM_DOWNLOAD_URL}/linux-${CUDA_ARCH_FOLDER}/${cuquantum_archive}"
    mkdir -p "${CUQUANTUM_INSTALL_PREFIX}" 
    tar xf "${cuquantum_archive}" --strip-components 1 -C "${CUQUANTUM_INSTALL_PREFIX}" 
    rm -rf "${cuquantum_archive}"
# [<cuQuantumInstall]
fi

if [ "$1" == "install-cuquantum" ]; then
    CUDA_VERSION=11.8 CUDA_ARCH_FOLDER=x86_64

# [>cuTensorInstall]
    CUTENSOR_VERSION=1.7.0.1
    CUTENSOR_DOWNLOAD_URL=https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor

    cutensor_archive=libcutensor-linux-${CUDA_ARCH_FOLDER}-${CUTENSOR_VERSION}-archive.tar.xz
    wget "${CUTENSOR_DOWNLOAD_URL}/linux-${CUDA_ARCH_FOLDER}/${cutensor_archive}"
    mkdir -p "${CUTENSOR_INSTALL_PREFIX}" && tar xf "${cutensor_archive}" --strip-components 1 -C "${CUTENSOR_INSTALL_PREFIX}"
    mv "${CUTENSOR_INSTALL_PREFIX}"/lib/$(echo ${CUDA_VERSION} | cut -d . -f1)/* ${CUTENSOR_INSTALL_PREFIX}/lib/
    ls -d ${CUTENSOR_INSTALL_PREFIX}/lib/*/ | xargs rm -rf && rm -rf "${cutensor_archive}"
# [<cuTensorInstall]
fi

if [ "$1" == "install-gcc" ]; then
# [>gccInstall]
    GCC_VERSION=11
    dnf install -y --nobest --setopt=install_weak_deps=False \
        gcc-toolset-${GCC_VERSION}
# [<gccInstall]
fi

# [>ToolchainConfiguration]
GCC_INSTALL_PREFIX=/opt/rh/gcc-toolset-11
export CXX="${GCC_INSTALL_PREFIX}/root/usr/bin/g++"
export CC="${GCC_INSTALL_PREFIX}/root/usr/bin/gcc"
export FC="${GCC_INSTALL_PREFIX}/root/usr/bin/gfortran"
export CUDACXX=/usr/local/cuda/bin/nvcc
# [>ToolchainConfiguration]

if [ "$1" == "install-prereqs" ]; then
    LLVM_BUILD_LINKER_FLAGS="-static-libgcc -static-libstdc++"
    export CMAKE_EXE_LINKER_FLAGS="$LLVM_BUILD_LINKER_FLAGS"
    export CMAKE_SHARED_LINKER_FLAGS="$LLVM_BUILD_LINKER_FLAGS"

    this_file_dir=`dirname "$(readlink -f "${BASH_SOURCE[0]}")"`
    source "$this_file_dir/install_prerequisites.sh"
fi

