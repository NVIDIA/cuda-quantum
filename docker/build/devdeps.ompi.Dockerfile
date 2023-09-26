# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds the necessary Open MPI dependencies used by CUDA Quantum.
# This image can be passed as argument to docker/build/devdeps.ext.Dockerfile
# to create a complete development environment for CUDA Quantum that contains all
# necessary dependencies.
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum-devdeps:ompi -f docker/build/devdeps.ompi.Dockerfile .

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

ARG TARGETARCH
ENV CUDA_INSTALL_PREFIX=/usr/local/cuda-11.8
ENV COMMON_COMPILER_FLAGS="-march=x86-64-v3 -mtune=generic -O2 -pipe"
ENV COMMON_COMPILER_FLAGS_ARM="-march=native -mtune=generic -O2 -pipe"

# 1 - Install basic tools needed for the builds

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ gfortran python3 python3-pip \
        libcurl4-openssl-dev libssl-dev liblapack-dev libpython3-dev \
        bzip2 make sudo vim curl git wget \
    && pip install --no-cache-dir numpy \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# 2 - Install SLURM PMI2 version 21.08.8

ENV PMI_INSTALL_PREFIX=/usr/local/pmi
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://download.schedmd.com/slurm/slurm-21.08.8.tar.bz2 \
    && tar -x -f /var/tmp/slurm-21.08.8.tar.bz2 -C /var/tmp -j && cd /var/tmp/slurm-21.08.8 \
    && export common_flags=$([ "$TARGETARCH" == "arm64" ] && echo "$COMMON_COMPILER_FLAGS_ARM" || echo "$COMMON_COMPILER_FLAGS") \
    &&  CC=gcc CFLAGS="$common_flags" \
        CXX=g++ CXXFLAGS="$common_flags" \
        F77=gfortran F90=gfortran FFLAGS="$common_flags" \
        FC=gfortran FCFLAGS="$common_flags" \
        LDFLAGS=-Wl,--as-needed \
        ./configure --prefix="$PMI_INSTALL_PREFIX" \
    && make -C contribs/pmi2 install \
    && rm -rf /var/tmp/slurm-21.08.8 /var/tmp/slurm-21.08.8.tar.bz2

# 3 - Install Mellanox OFED version 5.3-1.0.0.1

RUN wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add - \
    && mkdir -p /etc/apt/sources.list.d && wget -q -nc --no-check-certificate -P /etc/apt/sources.list.d https://linux.mellanox.com/public/repo/mlnx_ofed/5.3-1.0.0.1/ubuntu20.04/mellanox_mlnx_ofed.list \
    && apt-get update -y && apt-get install -y --no-install-recommends \
        ibverbs-providers ibverbs-utils \
        libibmad-dev libibmad5 libibumad-dev libibumad3 \
        libibverbs-dev libibverbs1 \
        librdmacm-dev librdmacm1 \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# 4 - Install GDRCOPY version 2.3.1

ENV GDRCOPY_VERSION=2.3.1 
ENV GDRCOPY_INSTALL_PREFIX=/usr/local/gdrcopy
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        autoconf automake \
        libgcrypt20-dev libnuma-dev libtool \
    && mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/NVIDIA/gdrcopy/archive/v${GDRCOPY_VERSION}.tar.gz \
    && tar -x -f /var/tmp/v${GDRCOPY_VERSION}.tar.gz -C /var/tmp -z && cd /var/tmp/gdrcopy-${GDRCOPY_VERSION} \
    && mkdir -p "$GDRCOPY_INSTALL_PREFIX/include" "$GDRCOPY_INSTALL_PREFIX/lib64" \
    && make PREFIX="$GDRCOPY_INSTALL_PREFIX" lib lib_install \
    && echo "$GDRCOPY_INSTALL_PREFIX/lib64" >> /etc/ld.so.conf.d/hpccm.conf && ldconfig \
    && rm -rf /var/tmp/gdrcopy-${GDRCOPY_VERSION} /var/tmp/v${GDRCOPY_VERSION}.tar.gz \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

ENV CPATH="$GDRCOPY_INSTALL_PREFIX/include:$CPATH"
ENV LIBRARY_PATH="$GDRCOPY_INSTALL_PREFIX/lib64:$LIBRARY_PATH"

# 5 - Install UCX version v1.13.1

ENV UCX_INSTALL_PREFIX=/usr/local/ucx
RUN mkdir -p /var/tmp && cd /var/tmp \
    && git clone https://github.com/openucx/ucx.git ucx && cd /var/tmp/ucx \
    && git checkout v1.13.1 \
    && ./autogen.sh \
    && export common_flags=$([ "$TARGETARCH" == "arm64" ] && echo "$COMMON_COMPILER_FLAGS_ARM" || echo "$COMMON_COMPILER_FLAGS") \
    &&  CC=gcc CFLAGS="$common_flags" \
        CXX=g++ CXXFLAGS="$common_flags" \
        F77=gfortran F90=gfortran FFLAGS="$common_flags" \
        FC=gfortran FCFLAGS="$common_flags" \
        LDFLAGS=-Wl,--as-needed \
        ./configure --prefix="$UCX_INSTALL_PREFIX" \
            --with-cuda="$CUDA_INSTALL_PREFIX" --with-gdrcopy="$GDRCOPY_INSTALL_PREFIX" \
            --disable-assertions --disable-backtrace-detail --disable-debug \
            --disable-params-check --disable-static \
            --disable-doxygen-doc --disable-logging \
            --enable-mt \
    && make -j$(nproc) && make -j$(nproc) install \
    && rm -rf /var/tmp/ucx

# 6 - Install MUNGE version 0.5.14

ENV MUNGE_INSTALL_PREFIX=/usr/local/munge
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/dun/munge/releases/download/munge-0.5.14/munge-0.5.14.tar.xz \
    && tar -x -f /var/tmp/munge-0.5.14.tar.xz -C /var/tmp -J && cd /var/tmp/munge-0.5.14 \
    && export common_flags=$([ "$TARGETARCH" == "arm64" ] && echo "$COMMON_COMPILER_FLAGS_ARM" || echo "$COMMON_COMPILER_FLAGS") \
    &&  CC=gcc CFLAGS="$common_flags" \
        CXX=g++ CXXFLAGS="$common_flags" \
        F77=gfortran F90=gfortran FFLAGS="$common_flags" \
        FC=gfortran FCFLAGS="$common_flags" \
        LDFLAGS=-Wl,--as-needed \
        ./configure --prefix="$MUNGE_INSTALL_PREFIX" \
    && make -j$(nproc) && make -j$(nproc) install \
    && rm -rf /var/tmp/munge-0.5.14 /var/tmp/munge-0.5.14.tar.xz

# 7 - Install PMIX version 3.2.3

ENV PMIX_INSTALL_PREFIX=/usr/local/pmix
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        hwloc libevent-dev \
    && mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/openpmix/openpmix/releases/download/v3.2.3/pmix-3.2.3.tar.gz \
    && tar -x -f /var/tmp/pmix-3.2.3.tar.gz -C /var/tmp -z && cd /var/tmp/pmix-3.2.3 \
    && export common_flags=$([ "$TARGETARCH" == "arm64" ] && echo "$COMMON_COMPILER_FLAGS_ARM" || echo "$COMMON_COMPILER_FLAGS") \
    &&  CC=gcc CFLAGS="$common_flags" \
        CXX=g++ CXXFLAGS="$common_flags" \
        F77=gfortran F90=gfortran FFLAGS="$common_flags" \
        FC=gfortran FCFLAGS="$common_flags" \
        LDFLAGS=-Wl,--as-needed \
        ./configure --prefix="$PMIX_INSTALL_PREFIX" \
            --with-munge="$MUNGE_INSTALL_PREFIX" \
    && make -j$(nproc) && make -j$(nproc) install \
    && rm -rf /var/tmp/pmix-3.2.3 /var/tmp/pmix-3.2.3.tar.gz \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

ENV CPATH="$PMIX_INSTALL_PREFIX/include:$CPATH" \
    LD_LIBRARY_PATH="$PMIX_INSTALL_PREFIX/lib:$LD_LIBRARY_PATH" \
    PATH="$PMIX_INSTALL_PREFIX/bin:$PATH"

# 8 - Install OMPI version 4.1.4

ENV OPENMPI_INSTALL_PREFIX=/usr/local/openmpi
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        flex openssh-client \
    && mkdir -p /var/tmp && cd /var/tmp \
    && git clone https://github.com/open-mpi/ompi.git ompi && cd /var/tmp/ompi \
    && git checkout v4.1.4 \
    && ./autogen.pl \
    && export common_flags=$([ "$TARGETARCH" == "arm64" ] && echo "$COMMON_COMPILER_FLAGS_ARM" || echo "$COMMON_COMPILER_FLAGS") \
    &&  CC=gcc CFLAGS="$common_flags" \
        CXX=g++ CXXFLAGS="$common_flags" \
        FC=gfortran FCFLAGS="$common_flags" \
        LDFLAGS=-Wl,--as-needed \
        ./configure --prefix="$OPENMPI_INSTALL_PREFIX" \
            --disable-getpwuid --disable-static \
            --disable-debug --disable-mem-debug --disable-mem-profile --disable-memchecker \
            --enable-mca-no-build=btl-uct --enable-mpi1-compatibility --enable-oshmem \
            --without-verbs \
            --with-cuda="$CUDA_INSTALL_PREFIX" \
            --with-slurm --with-pmi="$PMI_INSTALL_PREFIX" \
            --with-pmix="$PMIX_INSTALL_PREFIX" \
            --with-ucx="$UCX_INSTALL_PREFIX" \
    && make -j$(nproc) && make -j$(nproc) install \
    && rm -rf /var/tmp/ompi \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 
