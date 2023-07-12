# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This Dockerfile builds the most recent manylinux image for x86_64,
# then installs the dependencies needed on top of that for building
# the CUDA-Quantum pip wheel.

ARG manylinux_image=quay.io/pypa/manylinux_2_28_x86_64
FROM $manylinux_image
ARG llvm_commit

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive

# Install prerequisites for building LLVM.
RUN dnf check-update && dnf install -y --nobest --setopt=install_weak_deps=False \
        ninja-build cmake \
    && dnf clean all

# Clone the LLVM source code.
RUN mkdir /llvm-project && cd /llvm-project && git init \
    && git remote add origin https://github.com/llvm/llvm-project \
    && git fetch origin --depth=1 $llvm_commit && git reset --hard FETCH_HEAD

# Build the the LLVM libraries and compiler toolchain needed to build CUDA Quantum
ENV LLVM_INSTALL_PREFIX=/opt/llvm
ADD ./scripts/build_llvm.sh /scripts/build_llvm.sh
RUN export CMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++" && \
    export CMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++" \
    && bash /scripts/build_llvm.sh -s /llvm-project -c Release -v \
    && rm -rf /llvm-project && rm /scripts/build_llvm.sh

# Install additional dependencies required to build the CUDA Quantum wheel.
RUN dnf check-update && dnf install -y --nobest --setopt=install_weak_deps=False \
        glibc-static zlib-static perl-core \
    && dnf clean all
ENV OPENSSL_ROOT_DIR=/usr/local/ssl
RUN git clone https://github.com/openssl/openssl && cd openssl \
    && ./config --prefix="$OPENSSL_ROOT_DIR" --openssldir="$OPENSSL_ROOT_DIR" -static zlib \
    && make install && cd .. && rm -rf openssl
ENV BLAS_LIBRARIES=/usr/lib64/libblas.a
RUN dnf check-update && dnf install -y --nobest --setopt=install_weak_deps=False wget \
    && wget http://www.netlib.org/blas/blas-3.11.0.tgz \
    && tar -xzvf blas-3.11.0.tgz && cd BLAS-3.11.0 \
    && make && mv blas_LINUX.a "$BLAS_LIBRARIES" \
    && cd .. && rm -rf blas-3.11.0.tgz BLAS-3.11.0 \
    && dnf remove -y wget && dnf clean all