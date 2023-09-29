# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file contains additional CUDA Quantum development dependencies. 
# The image installs cuQuantum, cuTensor, and the CUDA packages defined by the
# cuda_packages build argument. It copies the OpenMPI installation and its 
# dependencies from the given ompidev_image. The copied paths can be configured
# via build arguments.  
#
# Usage:
# Must be built from the repo root with:
#   docker build -t ghcr.io/nvidia/cuda-quantum-devdeps:ext -f docker/build/devdeps.ext.Dockerfile .

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:llvm-main
ARG ompidev_image=ghcr.io/nvidia/cuda-quantum-devdeps:ompi-main
FROM $ompidev_image as ompibuild

FROM $base_image
SHELL ["/bin/bash", "-c"]

# When a dialogue box would be needed during install, assume default configurations.
# Set here to avoid setting it for all install commands. 
# Given as arg to make sure that this value is only set during build but not in the launched container.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt-get install -y --no-install-recommends ca-certificates wget \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Install Mellanox OFED runtime dependencies.

RUN apt-get update && apt-get install -y --no-install-recommends gnupg \
    && wget -qO - "https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox" | apt-key add - \
    && mkdir -p /etc/apt/sources.list.d && wget -q -nc --no-check-certificate -P /etc/apt/sources.list.d "https://linux.mellanox.com/public/repo/mlnx_ofed/5.3-1.0.0.1/ubuntu20.04/mellanox_mlnx_ofed.list" \
    && apt-get update -y && apt-get install -y --no-install-recommends \
        ibverbs-providers ibverbs-utils \
        libibmad5 libibumad3 libibverbs1 librdmacm1 \
    && apt-get remove -y gnupg \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Copy over SLURM PMI2.

ARG PMI_INSTALL_PREFIX=/usr/local/pmi
ENV PMI_INSTALL_PREFIX="$PMI_INSTALL_PREFIX"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PMIX_INSTALL_PREFIX/lib"
COPY --from=ompibuild "$PMI_INSTALL_PREFIX" "$PMI_INSTALL_PREFIX"

# Copy over GDRCOPY and install runtime dependencies.

ARG GDRCOPY_INSTALL_PREFIX=/usr/local/gdrcopy
ENV GDRCOPY_INSTALL_PREFIX="$GDRCOPY_INSTALL_PREFIX"
ENV CPATH="$GDRCOPY_INSTALL_PREFIX/include:$CPATH"
ENV LIBRARY_PATH="$GDRCOPY_INSTALL_PREFIX/lib64:$LIBRARY_PATH"
COPY --from=ompibuild "$GDRCOPY_INSTALL_PREFIX" "$GDRCOPY_INSTALL_PREFIX"

RUN echo "$GDRCOPY_INSTALL_PREFIX/lib64" >> /etc/ld.so.conf.d/hpccm.conf && ldconfig \
    && apt-get update -y && apt-get install -y --no-install-recommends \
        libgcrypt20 libnuma1 \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Copy over UCX.

ARG UCX_INSTALL_PREFIX=/usr/local/ucx
ENV UCX_INSTALL_PREFIX="$UCX_INSTALL_PREFIX"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$UCX_INSTALL_PREFIX/lib"
COPY --from=ompibuild "$UCX_INSTALL_PREFIX" "$UCX_INSTALL_PREFIX"

# Copy over MUNGE.

ARG MUNGE_INSTALL_PREFIX=/usr/local/munge
ENV MUNGE_INSTALL_PREFIX="$MUNGE_INSTALL_PREFIX"
COPY --from=ompibuild "$MUNGE_INSTALL_PREFIX" "$MUNGE_INSTALL_PREFIX"

# Copy over PMIX and install runtime dependencies.

ARG PMIX_INSTALL_PREFIX=/usr/local/pmix
ENV PMIX_INSTALL_PREFIX="$PMIX_INSTALL_PREFIX"
ENV PATH="$PMIX_INSTALL_PREFIX/bin:$PATH"
ENV CPATH="$PMIX_INSTALL_PREFIX/include:$CPATH"
ENV LD_LIBRARY_PATH="$PMIX_INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"
COPY --from=ompibuild "$PMIX_INSTALL_PREFIX" "$PMIX_INSTALL_PREFIX"

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        hwloc libevent-dev \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Copy over OpenMPI and install runtime dependencies.

ARG OPENMPI_INSTALL_PREFIX=/usr/local/openmpi
ENV OPENMPI_INSTALL_PREFIX="$OPENMPI_INSTALL_PREFIX"
ENV MPI_HOME="$OPENMPI_INSTALL_PREFIX"
ENV MPI_ROOT="$OPENMPI_INSTALL_PREFIX"
ENV PATH="$OPENMPI_INSTALL_PREFIX/bin:$PATH"
ENV CPATH="$OPENMPI_INSTALL_PREFIX/include:/usr/local/ofed/5.0-0/include:$CPATH"
ENV LIBRARY_PATH="/usr/local/ofed/5.0-0/lib:$LIBRARY_PATH"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$OPENMPI_INSTALL_PREFIX/lib"
COPY --from=ompibuild "$OPENMPI_INSTALL_PREFIX" "$OPENMPI_INSTALL_PREFIX"

RUN echo "$OPENMPI_INSTALL_PREFIX/lib" >> /etc/ld.so.conf.d/hpccm.conf && ldconfig \
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

ARG CUQUANTUM_INSTALL_PREFIX=/opt/nvidia/cuquantum
ENV CUQUANTUM_INSTALL_PREFIX="$CUQUANTUM_INSTALL_PREFIX"
ENV CUQUANTUM_ROOT="$CUQUANTUM_INSTALL_PREFIX"
ENV LD_LIBRARY_PATH="$CUQUANTUM_INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"
ENV CPATH="$CUQUANTUM_INSTALL_PREFIX/include:$CPATH"

ENV CUQUANTUM_VERSION=23.06.0.7_cuda11
RUN apt-get update && apt-get install -y --no-install-recommends xz-utils \
    && arch_folder=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64) \
    && wget -q "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-$arch_folder/cuquantum-linux-$arch_folder-23.06.0.7_cuda11-archive.tar.xz" \
    && mkdir -p "$CUQUANTUM_INSTALL_PREFIX" && tar xf cuquantum-linux-$arch_folder-$CUQUANTUM_VERSION-archive.tar.xz --strip-components 1 -C "$CUQUANTUM_INSTALL_PREFIX" \
    && rm cuquantum-linux-$arch_folder-$CUQUANTUM_VERSION-archive.tar.xz \
    && apt-get remove -y xz-utils \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Install cuTensor libraries.

ARG CUTENSOR_INSTALL_PREFIX=/opt/nvidia/cutensor
ENV CUTENSOR_INSTALL_PREFIX="$CUTENSOR_INSTALL_PREFIX"
ENV CUTENSOR_ROOT="$CUTENSOR_INSTALL_PREFIX"
ENV LD_LIBRARY_PATH="$CUTENSOR_INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"
ENV CPATH="$CUTENSOR_INSTALL_PREFIX/include:$CPATH"

ENV CUTENSOR_VERSION=1.7.0.1
RUN apt-get update && apt-get install -y --no-install-recommends xz-utils \
    && arch_folder=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64) \
    && wget -q "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-$arch_folder/libcutensor-linux-$arch_folder-$CUTENSOR_VERSION-archive.tar.xz" \
    && tar xf libcutensor-linux-$arch_folder-$CUTENSOR_VERSION-archive.tar.xz && cd libcutensor-linux-$arch_folder-$CUTENSOR_VERSION-archive \
    && mkdir -p "$CUTENSOR_INSTALL_PREFIX" && mv include "$CUTENSOR_INSTALL_PREFIX" && mv lib/11 "$CUTENSOR_INSTALL_PREFIX/lib" \
    && cd / && rm -rf libcutensor-linux-$arch_folder-$CUTENSOR_VERSION-archive* \
    && apt-get remove -y xz-utils \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* 

# Install CUDA 11.8.

ARG cuda_root=/usr/local/cuda-11.8
ARG cuda_packages="cuda-cudart-11-8 cuda-compiler-11-8 libcublas-dev-11-8"
RUN if [ -n "$cuda_packages" ]; then \
        arch_folder=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64) \
        && wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/$arch_folder/cuda-keyring_1.0-1_all.deb" \
        && dpkg -i cuda-keyring_1.0-1_all.deb \
        && apt-get update && apt-get install -y --no-install-recommends $cuda_packages \
        && rm cuda-keyring_1.0-1_all.deb \
        && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# The installation of CUDA above creates files that will be injected upon launching the container
# with the --gpu=all flag. This creates issues upon container launch. We hence remove these files.
# As long as the container is launched with the --gpu=all flag, the GPUs remain accessible and CUDA
# is fully functional. See also https://github.com/NVIDIA/nvidia-docker/issues/1699.
RUN if [ -z "$CUDA_ROOT" ]; then \
        rm -rf \
        /usr/lib/$(uname -m)-linux-gnu/libcuda.so* \
        /usr/lib/$(uname -m)-linux-gnu/libnvcuvid.so* \
        /usr/lib/$(uname -m)-linux-gnu/libnvidia-*.so* \
        /usr/lib/firmware \
        /usr/local/cuda/compat/lib; \
    fi

ENV CUDA_INSTALL_PREFIX="$cuda_root"
ENV CUDA_HOME="$CUDA_INSTALL_PREFIX"
ENV CUDA_ROOT="$CUDA_INSTALL_PREFIX"
ENV CUDA_PATH="$CUDA_INSTALL_PREFIX"
ENV PATH="${CUDA_INSTALL_PREFIX}/lib64/:${CUDA_INSTALL_PREFIX}/bin:${PATH}"

