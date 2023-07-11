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

# Install prerequisites for building LLVM
RUN dnf -y install ninja-build && \ 
    python3.10 -m pip install cmake lit --user 

# Clone the LLVM source code
RUN mkdir /llvm-project && cd /llvm-project && git init \
    && git remote add origin https://github.com/llvm/llvm-project \
    && git fetch origin --depth=1 $llvm_commit && git reset --hard FETCH_HEAD

# Build and install LLVM.
ENV LLVM_INSTALL_PREFIX=/opt/llvm
ADD ./scripts/build_llvm.sh /scripts/build_llvm.sh
RUN export CMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++" && \
    export CMAKE_SHARED_LINKER_FLAGS="-static-libgcc -static-libstdc++" \
    && bash /scripts/build_llvm.sh -s /llvm-project -c Release -v \
    && rm -rf /llvm-project 
RUN rm /scripts/build_llvm.sh

# Build and install OpenSSL, and BLAS.
# openssh-clients?
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        wget glibc-static zlib-static perl-core
RUN git clone https://github.com/openssl/openssl \
    && cd openssl && ./config --prefix=/usr/local/ssl --openssldir=/usr/local/ssl -static zlib \
    && make install 
RUN wget http://www.netlib.org/blas/blas-3.11.0.tgz \
    && tar -xzvf blas-3.11.0.tgz && cd BLAS-3.11.0 \
    && make && mv blas_LINUX.a /usr/lib64/libblas.a 