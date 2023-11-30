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
ARG pybind11_commit
ARG toolchain=gcc11

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive

# Clone the LLVM source code.
# Preserve access to the history to be able to cherry pick specific commits.
RUN git clone --filter=tree:0 https://github.com/llvm/llvm-project /llvm-project \
    && cd /llvm-project && git checkout $llvm_commit

# Install the C/C++ compiler toolchain to build the LLVM dependencies.
# CUDA Quantum needs to be built with that same toolchain, and the
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

# Build pybind11 - 
# we should be able to use the same pybind version independent on what Python version we generate bindings for.
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        ninja-build cmake python3-devel \
    && mkdir /pybind11-project && cd /pybind11-project && git init \
    && git remote add origin https://github.com/pybind/pybind11 \
    && git fetch origin --depth=1 $pybind11_commit && git reset --hard FETCH_HEAD \
    && mkdir -p /pybind11-project/build && cd /pybind11-project/build \
    && python3 -m ensurepip --upgrade && python3 -m pip install pytest \
    && cmake -G Ninja ../ -DCMAKE_INSTALL_PREFIX=/usr/local/pybind11 -DPYTHON_EXECUTABLE="$(which python3)" \
    && cmake --build . --target install --config Release \
    && python3 -m pip uninstall -y pytest \
    && cd / && rm -rf /pybind11-project

# Build the the LLVM libraries and compiler toolchain needed to build CUDA Quantum.
ADD ./scripts/build_llvm.sh /scripts/build_llvm.sh
ENV LLVM_BUILD_LINKER_FLAGS="-static-libgcc -static-libstdc++"
RUN export CMAKE_EXE_LINKER_FLAGS="$LLVM_BUILD_LINKER_FLAGS" CMAKE_SHARED_LINKER_FLAGS="$LLVM_BUILD_LINKER_FLAGS" \
    && bash /scripts/build_llvm.sh -s /llvm-project -p "clang;lld;mlir" -c Release -v
    # No clean up of the build or source directory,
    # since we need to re-build llvm for each python version to get the bindings.

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
