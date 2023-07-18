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

ARG manylinux_image=quay.io/pypa/manylinux_2_28_x86_64:latest
FROM $manylinux_image

ARG llvm_commit
ENV CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
ENV CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive

# Clone the LLVM source code.
RUN mkdir /llvm-project && cd /llvm-project && git init \
    && git remote add origin https://github.com/llvm/llvm-project \
    && git fetch origin --depth=1 $llvm_commit && git reset --hard FETCH_HEAD

# Build the the LLVM libraries and compiler toolchain needed to build CUDA Quantum
ENV LLVM_INSTALL_PREFIX=/opt/llvm
ADD ./scripts/build_llvm.sh /scripts/build_llvm.sh
RUN dnf check-update && dnf install -y --nobest --setopt=install_weak_deps=False \
        ninja-build cmake \
    && export CMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++" \
    && export CMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++" \
    && bash /scripts/build_llvm.sh -s /llvm-project -c Release -v \
    && dnf remove -y ninja-build cmake && dnf clean all \
    && rm -rf /llvm-project && rm /scripts/build_llvm.sh

# Install additional dependencies required to build the CUDA Quantum wheel.
RUN dnf check-update && dnf install -y --nobest --setopt=install_weak_deps=False \
        glibc-static perl-core \
    && dnf clean all

# Build OpenBLAS from source with OpenMP enabled.
ADD ./scripts/install_prerequisites.sh /scripts/install_prerequisites.sh
ENV OPENBLAS_INSTALL_PREFIX=/usr/local/openblas
ENV OPENSSL_INSTALL_PREFIX=/usr/local/openssl
RUN dnf check-update && dnf install -y --nobest --setopt=install_weak_deps=False wget \
    && bash /scripts/install_prerequisites.sh \
    && dnf remove -y wget && dnf clean all
