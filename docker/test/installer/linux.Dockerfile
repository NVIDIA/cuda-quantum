# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=amd64/almalinux:8

# [OpenMPI Installation]
FROM amd64/almalinux:8 as mpibuild
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
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

# [CUDA Quantum Installation]
FROM $base_image
ARG base_image

SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ADD docker/test/installer/dependencies.sh /runtime_dependencies.sh
RUN bash runtime_dependencies.sh ${base_image}

## [Install]
ARG cuda_quantum_installer='cuda_quantum.*'
ADD "${cuda_quantum_installer}" .
RUN "$(ls "${cuda_quantum_installer}")" --accept

ENV CUDA_QUANTUM_PATH=/opt/nvidia/cudaq
ENV PATH="${CUDA_QUANTUM_PATH}/bin:${PATH}"
ENV CPLUS_INCLUDE_PATH="${CUDA_QUANTUM_PATH}/include:${CPLUS_INCLUDE_PATH}"

## [Enable MPI support]
COPY --from=mpibuild /usr/local/openmpi/ /usr/local/openmpi
RUN MPI_PATH=/usr/local/openmpi \
    bash "${CUDA_QUANTUM_PATH}/distributed_interfaces/activate_custom_mpi.sh"

## [Validate]
ADD scripts/validate_container.sh validate.sh
ADD docs/sphinx/examples/cpp examples
#RUN bash validate.sh
