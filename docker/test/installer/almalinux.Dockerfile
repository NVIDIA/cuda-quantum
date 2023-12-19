# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [OpenMPI Installation]
FROM amd64/almalinux:8 as mpibuild
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        'dnf-command(config-manager)'

## [Prerequisites]
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh
RUN source /cuda-quantum/scripts/configure_build.sh install-cuda
RUN source /cuda-quantum/scripts/configure_build.sh install-gcc
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        autoconf libtool flex make wget

## [Build]
RUN source /cuda-quantum/scripts/configure_build.sh build-openmpi

# [CUDA Quantum - Ubuntu]
FROM ubuntu:22.04 as ubuntu
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates

## [Runtime dependencies]
RUN apt-get install -y --no-install-recommends libstdc++-11-dev
RUN cuda_packages="cuda-cudart-11-8 cuda-nvtx-11-8 libcusolver-11-8 libcublas-11-8" && \
    if [ -n "$cuda_packages" ]; then \
        arch_folder=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64) && \
        wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/$arch_folder/cuda-keyring_1.0-1_all.deb" && \
        dpkg -i cuda-keyring_1.0-1_all.deb && \
        apt-get update && apt-get install -y --no-install-recommends $cuda_packages && \
        rm cuda-keyring_1.0-1_all.deb; \
    fi

## [Installation]
ADD out/cuda_quantum.* .
RUN ./cuda_quantum.$(uname -m) --accept

ENV CUDA_QUANTUM_PATH=/opt/nvidia/cudaq
ENV PATH="${CUDA_QUANTUM_PATH}/bin:${PATH}"
ENV CPLUS_INCLUDE_PATH="${CUDA_QUANTUM_PATH}/include:${CPLUS_INCLUDE_PATH}"

## [Enable MPI support]
COPY --from=mpibuild /usr/local/openmpi/ /usr/local/openmpi
RUN MPI_PATH=/usr/local/openmpi \
    bash "${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh"

## [Validation]
ADD scripts/validate_container.sh validate.sh
ADD docs/sphinx/examples examples
#RUN bash validate.sh

# [CUDA Quantum - AlmaLinux]
FROM amd64/almalinux:8 as almalinux
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        'dnf-command(config-manager)'

## [Runtime dependencies]
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh
RUN source /cuda-quantum/scripts/configure_build.sh install-cudart
#RUN source /cuda-quantum/scripts/configure_build.sh install-gcc

## [Installation]
ADD out/cuda_quantum.* .
RUN ./cuda_quantum.$(uname -m) --accept

ENV CUDA_QUANTUM_PATH=/opt/nvidia/cudaq
ENV PATH="${CUDA_QUANTUM_PATH}/bin:${PATH}"
ENV CPLUS_INCLUDE_PATH="${CUDA_QUANTUM_PATH}/include:${CPLUS_INCLUDE_PATH}"

## [Enable MPI support]
COPY --from=mpibuild /usr/local/openmpi/ /usr/local/openmpi
RUN MPI_PATH=/usr/local/openmpi \
    bash "${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh"

## [Validation]
ADD scripts/validate_container.sh validate.sh
ADD docs/sphinx/examples examples
#RUN bash validate.sh
