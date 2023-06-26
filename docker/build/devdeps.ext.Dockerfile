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
#   docker build -t ghcr.io/nvidia/cuda-quantum-devdeps:ext-${toolchain} -f docker/build/devdeps.ext.Dockerfile .
#
# The variable $toolchain should indicate which compiler toolchain the development environment 
# which this image extends is configure with; see also docker/build/devdeps.Dockerfile.

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:llvm-main
ARG ompidev_image=ghcr.io/nvidia/cuda-quantum-devdeps:ompi-main
FROM $ompidev_image as ompibuild

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

ARG cuda_packages="cuda-cudart-11-8 cuda-compiler-11-8 libcublas-dev-11-8"
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
