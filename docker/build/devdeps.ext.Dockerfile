# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file extends the CUDA Quantum development dependencies to include the necessary 
# dependencies for GPU components and backends. This image include an OpenMPI
# installation as well as the configured CUDA packages. Which CUDA packages are 
# included is defined by the cuda_packages argument. 
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum-devdeps:${toolchain}-ext -f docker/build/devdeps.ext.Dockerfile .
#
# The variable $toolchain should indicate which compiler toolchain the development environment 
# which this image extends is configure with; see also docker/build/devdeps.Dockerfile.

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:llvm-main

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as ompibuild
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_INSTALL_PREFIX=/usr/local/cuda-11.8
ENV COMMON_COMPILER_FLAGS="-march=x86-64-v3 -mtune=generic -O2 -pipe"

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
    &&  CC=gcc CFLAGS="$COMMON_COMPILER_FLAGS" \
        CXX=g++ CXXFLAGS="$COMMON_COMPILER_FLAGS" \
        F77=gfortran F90=gfortran FFLAGS="$COMMON_COMPILER_FLAGS" \
        FC=gfortran FCFLAGS="$COMMON_COMPILER_FLAGS" \
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

# 4 - Install GDRCOPY version 2.1

ENV GDRCOPY_INSTALL_PREFIX=/usr/local/gdrcopy
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        autoconf automake \
        libgcrypt20-dev libnuma-dev libtool \
    && mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://github.com/NVIDIA/gdrcopy/archive/v2.1.tar.gz \
    && tar -x -f /var/tmp/v2.1.tar.gz -C /var/tmp -z && cd /var/tmp/gdrcopy-2.1 \
    && mkdir -p "$GDRCOPY_INSTALL_PREFIX/include" "$GDRCOPY_INSTALL_PREFIX/lib64" \
    && make PREFIX="$GDRCOPY_INSTALL_PREFIX" lib lib_install \
    && echo "$GDRCOPY_INSTALL_PREFIX/lib64" >> /etc/ld.so.conf.d/hpccm.conf && ldconfig \
    && rm -rf /var/tmp/gdrcopy-2.1 /var/tmp/v2.1.tar.gz \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

ENV CPATH="$GDRCOPY_INSTALL_PREFIX/include:$CPATH"
ENV LIBRARY_PATH="$GDRCOPY_INSTALL_PREFIX/lib64:$LIBRARY_PATH"

# 5 - Install UCX version v1.13.1

ENV UCX_INSTALL_PREFIX=/usr/local/ucx
RUN mkdir -p /var/tmp && cd /var/tmp \
    && git clone https://github.com/openucx/ucx.git ucx && cd /var/tmp/ucx \
    && git checkout v1.13.1 \
    && ./autogen.sh \
    &&  CC=gcc CFLAGS="$COMMON_COMPILER_FLAGS" \
        CXX=g++ CXXFLAGS="$COMMON_COMPILER_FLAGS" \
        F77=gfortran F90=gfortran FFLAGS="$COMMON_COMPILER_FLAGS" \
        FC=gfortran FCFLAGS="$COMMON_COMPILER_FLAGS" \
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
    &&  CC=gcc CFLAGS="$COMMON_COMPILER_FLAGS" \
        CXX=g++ CXXFLAGS="$COMMON_COMPILER_FLAGS" \
        F77=gfortran F90=gfortran FFLAGS="$COMMON_COMPILER_FLAGS" \
        FC=gfortran FCFLAGS="$COMMON_COMPILER_FLAGS" \
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
    &&  CC=gcc CFLAGS="$COMMON_COMPILER_FLAGS" \
        CXX=g++ CXXFLAGS="$COMMON_COMPILER_FLAGS" \
        F77=gfortran F90=gfortran FFLAGS="$COMMON_COMPILER_FLAGS" \
        FC=gfortran FCFLAGS="$COMMON_COMPILER_FLAGS" \
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
    &&  CC=gcc CFLAGS="$COMMON_COMPILER_FLAGS" \
        CXX=g++ CXXFLAGS="$COMMON_COMPILER_FLAGS" \
        F77=gfortran F90=gfortran FFLAGS="$COMMON_COMPILER_FLAGS" \
        FC=gfortran FCFLAGS="$COMMON_COMPILER_FLAGS" \
        LDFLAGS=-Wl,--as-needed \
        ./configure --prefix="$OPENMPI_INSTALL_PREFIX" \
            --disable-getpwuid --disable-static \
            --disable-debug --disable-mem-debug --disable-mem-profile --disable-memchecker \
            --enable-mca-no-build=btl-uct --enable-mpi1-compatibility --enable-oshmem \
            --with-cuda="$CUDA_INSTALL_PREFIX" \
            --with-slurm --with-pmi="$PMI_INSTALL_PREFIX" \
            --with-pmix="$PMIX_INSTALL_PREFIX" \
            --with-ucx="$UCX_INSTALL_PREFIX" \
            --without-verbs \
    && make -j$(nproc) && make -j$(nproc) install \
    && rm -rf /var/tmp/ompi \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Build the final image that has CUDA Quantum and all its dev dependencies installed, as well as
# OpenMPI, its dependencies, and additional tools for developing CUDA Quantum backends and extensions.
FROM $base_image
SHELL ["/bin/bash", "-c"]

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt-get install -y --no-install-recommends \
        ca-certificates openssl wget \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Install Mellanox OFED runtime dependencies.

RUN apt-get update && apt-get install -y --no-install-recommends gnupg \
    && wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add - \
    && mkdir -p /etc/apt/sources.list.d && wget -q -nc --no-check-certificate -P /etc/apt/sources.list.d https://linux.mellanox.com/public/repo/mlnx_ofed/5.3-1.0.0.1/ubuntu20.04/mellanox_mlnx_ofed.list \
    && apt-get update -y && apt-get install -y --no-install-recommends \
        ibverbs-providers ibverbs-utils \
        libibmad5 libibumad3 libibverbs1 librdmacm1 \
    && apt-get remove -y gnupg \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Copy over SLURM PMI2.

COPY --from=ompibuild /usr/local/pmi /usr/local/pmi
ENV PMI_INSTALL_PREFIX=/usr/local/pmi
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PMIX_INSTALL_PREFIX/lib"

# Copy over GDRCOPY and install runtime dependencies.

COPY --from=ompibuild /usr/local/gdrcopy /usr/local/gdrcopy
ENV GDRCOPY_INSTALL_PREFIX=/usr/local/gdrcopy
ENV CPATH="$GDRCOPY_INSTALL_PREFIX/include:$CPATH"
ENV LIBRARY_PATH="$GDRCOPY_INSTALL_PREFIX/lib64:$LIBRARY_PATH"

RUN echo "$GDRCOPY_INSTALL_PREFIX/lib64" >> /etc/ld.so.conf.d/hpccm.conf && ldconfig \
    && apt-get update -y && apt-get install -y --no-install-recommends \
        libgcrypt20 libnuma1 \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Copy over UCX.

COPY --from=ompibuild /usr/local/ucx /usr/local/ucx
ENV UCX_INSTALL_PREFIX=/usr/local/ucx
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$UCX_INSTALL_PREFIX/lib"

# Copy over MUNGE.

COPY --from=ompibuild /usr/local/munge /usr/local/munge
ENV MUNGE_INSTALL_PREFIX=/usr/local/munge

# Copy over PMIX and install runtime dependencies.

COPY --from=ompibuild /usr/local/pmix /usr/local/pmix
ENV PMIX_INSTALL_PREFIX=/usr/local/pmix
ENV PATH="$PMIX_INSTALL_PREFIX/bin:$PATH"
ENV CPATH="$PMIX_INSTALL_PREFIX/include:$CPATH"
ENV LD_LIBRARY_PATH="$PMIX_INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        hwloc libevent-dev \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Copy over OpenMPI and install runtime dependencies.

COPY --from=ompibuild /usr/local/openmpi /usr/local/openmpi
ENV OPENMPI_INSTALL_PREFIX=/usr/local/openmpi
ENV MPI_HOME="$OPENMPI_INSTALL_PREFIX"
ENV MPI_ROOT="$OPENMPI_INSTALL_PREFIX"
ENV PATH="$PATH:$OPENMPI_INSTALL_PREFIX/bin"
ENV CPATH="$OPENMPI_INSTALL_PREFIX/include:/usr/local/ofed/5.0-0/include:$CPATH"
ENV LIBRARY_PATH="/usr/local/ofed/5.0-0/lib:$LIBRARY_PATH"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$OPENMPI_INSTALL_PREFIX/lib"

RUN echo "/usr/local/openmpi/lib" >> /etc/ld.so.conf.d/hpccm.conf && ldconfig \
    && apt-get update -y && apt-get install -y --no-install-recommends \
        flex openssh-client \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Set some configurations in the form of environment variables.

ENV OMPI_MCA_btl=^smcuda,vader,tcp,uct,openib
ENV OMPI_MCA_pml=ucx
ENV UCX_IB_PCI_RELAXED_ORDERING=on
ENV UCX_MAX_RNDV_RAILS=1
ENV UCX_MEMTYPE_CACHE=n
ENV UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy,sm

# Install cuQuantum libraries.

RUN apt-get update && apt-get install -y --no-install-recommends xz-utils \
    && wget https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-22.11.0.13-archive.tar.xz \
    && tar xf cuquantum-linux-x86_64-22.11.0.13-archive.tar.xz \
    && mkdir -p /opt/nvidia && mv cuquantum-linux-x86_64-22.11.0.13-archive /opt/nvidia/cuquantum \
    && cd / && rm -rf cuquantum-linux-x86_64-22.11.0.13-archive.tar.xz \
    && apt-get remove -y xz-utils \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

ENV CUQUANTUM_INSTALL_PREFIX=/opt/nvidia/cuquantum
ENV LD_LIBRARY_PATH="$CUQUANTUM_INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"

# Install cuTensor libraries.
RUN apt-get update && apt-get install -y --no-install-recommends xz-utils \
    && wget https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-1.6.2.3-archive.tar.xz \
    && tar xf libcutensor-linux-x86_64-1.6.2.3-archive.tar.xz && cd libcutensor-linux-x86_64-1.6.2.3-archive \
    && mkdir -p /opt/nvidia/cutensor && mv include /opt/nvidia/cutensor/ && mv lib/11 /opt/nvidia/cutensor/lib \
    && cd / && rm -rf libcutensor-linux-x86_64-1.6.2.3-archive* \
    && apt-get remove -y xz-utils \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

ENV CUTENSOR_INSTALL_PREFIX=/opt/nvidia/cutensor
ENV LD_LIBRARY_PATH="$CUTENSOR_INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"

# Install CUDA 11.8.

# cuda-compiler-11-8 cuda-cudart-11-8 cuda-cccl-11-8
# cuda-cudart-dev-11-8
# cuda-command-line-tools-11-8n
ARG cuda_packages="cuda-cudart-11-8 cuda-compiler-11-8"
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends $cuda_packages \
    && rm cuda-keyring_1.0-1_all.deb \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# The installation of CUDA above creates files that will be injected upon launching the container
# with the --gpu=all flag. This creates issues upon container launch. We hence remove these files.
# As long as the container is launched with the --gpu=all flag, the GPUs remain accessible and CUDA
# is fully functional. See also https://github.com/NVIDIA/nvidia-docker/issues/1699.
RUN rm -rf \
    /usr/lib/x86_64-linux-gnu/libcuda.so* \
    /usr/lib/x86_64-linux-gnu/libnvcuvid.so* \
    /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
    /usr/lib/firmware \
    /usr/local/cuda/compat/lib

ENV CUDA_INSTALL_PREFIX=/usr/local/cuda-11.8
ENV CUDA_HOME="$CUDA_INSTALL_PREFIX"
ENV CUDA_ROOT="$CUDA_INSTALL_PREFIX"
ENV CUDA_PATH="$CUDA_INSTALL_PREFIX"
ENV PATH="${CUDA_INSTALL_PREFIX}/lib64/:${PATH}:${CUDA_INSTALL_PREFIX}/bin"
ENV LD_LIBRARY_PATH="${CUDA_INSTALL_PREFIX}/lib64:${CUDA_INSTALL_PREFIX}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
