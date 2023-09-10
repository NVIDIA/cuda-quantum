# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds the development environment that contains the necessary development 
# dependencies for building a CUDA Quantum Python wheel. It does not include the CUDA,
# OpenMPI and other dependencies that some of the simulator backends require.
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum-devdeps:manylinux -f docker/build/devdeps.manylinux.Dockerfile .
#
# The variable $toolchain indicates which compiler toolchain to build the LLVM libraries with. 
# The toolchain used to build the LLVM binaries that CUDA Quantum depends on must be used to build
# CUDA Quantum. This image sets the CC and CXX environment variables to use that toolchain. 
# Currently, clang15 and gcc11 are supported.

# There are currently no multi-platform manylinux images available.
# See https://github.com/pypa/manylinux/issues/1306.
ARG arch=x86_64
ARG manylinux_image=manylinux_2_28
FROM quay.io/pypa/${manylinux_image}_${arch}:latest

ARG distro=rhel8
ARG llvm_commit
ARG toolchain=gcc11

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive

# Clone the LLVM source code.
RUN mkdir /llvm-project && cd /llvm-project && git init \
    && git remote add origin https://github.com/llvm/llvm-project \
    && git fetch origin --depth=1 $llvm_commit && git reset --hard FETCH_HEAD

# Install the C/C++ compiler toolchain with which the LLVM dependencies have
# been built. CUDA Quantum needs to be built with that same toolchain, and the
# toolchain needs to be one of the supported CUDA host compilers. We use
# a wrapper script so that the path that we set CC and CXX to is independent 
# on the installed toolchain. Unfortunately, a symbolic link won't work.
# Using update-alternatives for c++ and cc could maybe be a better option.
ENV LLVM_INSTALL_PREFIX=/opt/llvm
RUN if [ "$toolchain" == 'gcc11' ]; then \
        dev_tools=gcc-toolset-11 && CC=$(which gcc | sed 's/[0-9]\{1,2\}/11/g') && CXX=$(which g++ | sed 's/[0-9]\{1,2\}/11/g'); \
    elif [ "$toolchain" == 'clang15' ]; then \
        dev_tools=clang && CC=$(which clang-15) && CXX=$(which clang++-15); \
    else echo "Toolchain not supported." && exit 1; \
    fi \
    && dnf install -y --nobest --setopt=install_weak_deps=False $dev_tools.$(uname -m) \
    && dnf clean all \
    && mkdir -p "$LLVM_INSTALL_PREFIX/bootstrap" \
    && echo -e '#!/bin/bash\n"'$CC'" "$@"' > "$LLVM_INSTALL_PREFIX/bootstrap/cc" \
    && echo -e '#!/bin/bash\n"'$CXX'" "$@"' > "$LLVM_INSTALL_PREFIX/bootstrap/cxx" \
    && chmod +x "$LLVM_INSTALL_PREFIX/bootstrap/cc" \
    && chmod +x "$LLVM_INSTALL_PREFIX/bootstrap/cxx"
ENV CC="$LLVM_INSTALL_PREFIX/bootstrap/cc"
ENV CXX="$LLVM_INSTALL_PREFIX/bootstrap/cxx"

# Build the the LLVM libraries and compiler toolchain needed to build CUDA Quantum
ADD ./scripts/build_llvm.sh /scripts/build_llvm.sh
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        ninja-build cmake \
    && export CMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++" \
    && export CMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++" \
    && bash /scripts/build_llvm.sh -s /llvm-project -c Release -v \
    && dnf remove -y ninja-build cmake && dnf clean all \
    && rm -rf /llvm-project && rm /scripts/build_llvm.sh

# Install additional dependencies required to build the CUDA Quantum wheel.
ADD ./scripts/install_prerequisites.sh /scripts/install_prerequisites.sh
ENV BLAS_INSTALL_PREFIX=/usr/local/blas
ENV OPENSSL_INSTALL_PREFIX=/usr/local/openssl
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        glibc-static perl-core wget cmake \
    && bash /scripts/install_prerequisites.sh \
    && dnf remove -y wget cmake && dnf clean all \
    && rm -rf /scripts/install_prerequisites.sh

# Install CUDA 11.8.

# Note that pip packages are available for all necessary runtime components.
RUN arch_folder=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64) \
    && dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch_folder/cuda-$distro.repo \
    && dnf clean expire-cache \
    && dnf install -y --nobest --setopt=install_weak_deps=False \
        cuda-compiler-11-8.$(uname -m) cuda-cudart-devel-11-8.$(uname -m) libcublas-devel-11-8.$(uname -m)

ENV CUDA_INSTALL_PREFIX=/usr/local/cuda-11.8
ENV CUDA_HOME="$CUDA_INSTALL_PREFIX"
ENV CUDA_ROOT="$CUDA_INSTALL_PREFIX"
ENV CUDA_PATH="$CUDA_INSTALL_PREFIX"
ENV PATH="${CUDA_INSTALL_PREFIX}/lib64/:${CUDA_INSTALL_PREFIX}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_INSTALL_PREFIX}/lib64:${LD_LIBRARY_PATH}"
