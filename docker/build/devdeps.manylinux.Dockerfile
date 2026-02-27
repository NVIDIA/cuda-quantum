# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds the development environment that contains the necessary development 
# dependencies for building a CUDA-Q Python wheel.
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum-devdeps:manylinux -f docker/build/devdeps.manylinux.Dockerfile .
#
# The variable $toolchain indicates which compiler toolchain to build the LLVM libraries with. 
# The toolchain used to build the LLVM binaries that CUDA-Q depends on must be used to build
# CUDA-Q. This image sets the CC and CXX environment variables to use that toolchain. 
# Currently, clang16 and gcc11, gcc12, and gcc13 are supported.

# There are currently no multi-platform manylinux images available.
# See https://github.com/pypa/manylinux/issues/1306.
ARG base_image=quay.io/pypa/manylinux_2_28_x86_64:latest
FROM ${base_image}

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
# CUDA-Q needs to be built with that same toolchain, and the
# toolchain needs to be one of the supported CUDA host compilers. We use
# a wrapper script so that the path that we set CC and CXX to is independent 
# on the installed toolchain. Unfortunately, a symbolic link won't work.
# Using update-alternatives for c++ and cc could maybe be a better option.
ENV LLVM_INSTALL_PREFIX=/usr/local/llvm
RUN if [ "${toolchain#gcc}" != "$toolchain" ]; then \
        gcc_version=`echo $toolchain | grep -o '[0-9]*'` && \
        if [ -z "$(which gcc 2> /dev/null | grep $gcc_version)" ]; then \
            # Using releasever=8.9: boost packages missing from 8.10 mirrors for aarch64
            dnf install -y --nobest --setopt=install_weak_deps=False --releasever=8.9 gcc-toolset-$gcc_version && \
            enable_script=`find / -path '*gcc*' -path '*'$gcc_version'*' -name enable` && . "$enable_script"; \
        fi && \
        CC="$(which gcc)" && CXX="$(which g++)"; \
    elif [ "$toolchain" == 'clang16' ]; then \
        dnf install -y --nobest --setopt=install_weak_deps=False clang-16.0.6 && \
        CC="$(which clang-16)" && CXX="$(which clang++-16)"; \
    else echo "Toolchain not supported." && exit 1; \
    fi && dnf clean all \
    && mkdir -p "$LLVM_INSTALL_PREFIX/bootstrap" \
    && echo -e '#!/bin/bash\n"'$CC'" "$@"' > "$LLVM_INSTALL_PREFIX/bootstrap/cc" \
    && echo -e '#!/bin/bash\n"'$CXX'" "$@"' > "$LLVM_INSTALL_PREFIX/bootstrap/cxx" \
    && chmod +x "$LLVM_INSTALL_PREFIX/bootstrap/cc" \
    && chmod +x "$LLVM_INSTALL_PREFIX/bootstrap/cxx"
ENV CC="$LLVM_INSTALL_PREFIX/bootstrap/cc"
ENV CXX="$LLVM_INSTALL_PREFIX/bootstrap/cxx"

# Build pybind11 - 
# we should be able to use the same pybind version independent on what Python version we generate bindings for.
ENV PYBIND11_INSTALL_PREFIX=/usr/local/pybind11
# Using releasever=8.9: cmake packages missing from 8.10 mirrors for aarch64
RUN dnf install -y --nobest --setopt=install_weak_deps=False --releasever=8.9\
        ninja-build cmake python3-devel \
    && mkdir /pybind11-project && cd /pybind11-project && git init \
    && git remote add origin https://github.com/pybind/pybind11 \
    && git fetch origin --depth=1 $pybind11_commit && git reset --hard FETCH_HEAD \
    && mkdir -p /pybind11-project/build && cd /pybind11-project/build \
    && cmake -G Ninja ../ -DCMAKE_INSTALL_PREFIX="$PYBIND11_INSTALL_PREFIX" -DPYTHON_EXECUTABLE="$(which python3)" -DPYBIND11_TEST=False \
    && cmake --build . --target install --config Release \
    && cd / && rm -rf /pybind11-project

RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.28.4/cmake-3.28.4-linux-$(uname -m).sh -o cmake-install.sh \
    && bash cmake-install.sh --skip-licence --exclude-subdir --prefix=/usr/local \
    && rm cmake-install.sh

# Build the the LLVM libraries and compiler toolchain needed to build CUDA-Q.
ADD ./scripts/build_llvm.sh /scripts/build_llvm.sh
ADD ./cmake/caches/LLVM.cmake /cmake/caches/LLVM.cmake
ADD ./tpls/customizations/llvm/ /tpls/customizations/llvm/
RUN LLVM_PROJECTS='clang;mlir' LLVM_SOURCE=/llvm-project \
    LLVM_CMAKE_CACHE=/cmake/caches/LLVM.cmake \
    LLVM_CMAKE_PATCHES=/tpls/customizations/llvm \
    bash /scripts/build_llvm.sh -c Release -v
    # No clean up of the build or source directory,
    # since we need to re-build llvm for each python version to get the bindings.

# Install CUDA

ARG cuda_version=12.6
ENV CUDA_VERSION=${cuda_version}

RUN arch_folder=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64) \
    && dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch_folder/cuda-$distro.repo \
    && dnf clean expire-cache \
    # Using releasever=8.9: cmake packages missing from 8.10 mirrors for aarch64
    && dnf install -y --nobest --setopt=install_weak_deps=False --releasever=8.9 wget \
        cuda-compiler-$(echo ${CUDA_VERSION} | tr . -) \
        cuda-cudart-devel-$(echo ${CUDA_VERSION} | tr . -) \
        libcublas-devel-$(echo ${CUDA_VERSION} | tr . -) \
        libcurand-devel-$(echo ${CUDA_VERSION} | tr . -) \
        libcusparse-devel-$(echo ${CUDA_VERSION} | tr . -)

ENV CUDA_INSTALL_PREFIX=/usr/local/cuda-$CUDA_VERSION
ENV CUDA_HOME="$CUDA_INSTALL_PREFIX"
ENV CUDA_ROOT="$CUDA_INSTALL_PREFIX"
ENV CUDA_PATH="$CUDA_INSTALL_PREFIX"
ENV PATH="${CUDA_INSTALL_PREFIX}/lib64/:${CUDA_INSTALL_PREFIX}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_INSTALL_PREFIX}/lib64:${LD_LIBRARY_PATH}"

# Install additional dependencies required to build the CUDA-Q wheel.
ADD ./scripts/install_prerequisites.sh /scripts/install_prerequisites.sh
ADD ./scripts/configure_build.sh /scripts/configure_build.sh
ENV BLAS_INSTALL_PREFIX=/usr/local/blas
ENV ZLIB_INSTALL_PREFIX=/usr/local/zlib
ENV OPENSSL_INSTALL_PREFIX=/usr/local/openssl
ENV CURL_INSTALL_PREFIX=/usr/local/curl
ENV AWS_INSTALL_PREFIX=/usr/local/aws
ENV CUQUANTUM_INSTALL_PREFIX=/usr/local/cuquantum
ENV CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor
RUN bash /scripts/install_prerequisites.sh
