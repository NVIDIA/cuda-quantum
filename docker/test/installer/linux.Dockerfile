# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=amd64/almalinux:8
ARG base_image_mpibuild=${base_image}

# [OpenMPI Installation]
FROM ${base_image_mpibuild} as mpibuild
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
FROM ${base_image}
ARG base_image

ARG DEBIAN_FRONTEND=noninteractive
ADD docker/test/installer/dependencies.sh /runtime_dependencies.sh
RUN bash runtime_dependencies.sh ${base_image}

## [MPI Installation]
COPY --from=mpibuild /usr/local/openmpi/ /usr/local/openmpi
RUN ln -s /usr/local/openmpi/bin/mpiexec /bin/mpiexec

# Create new user `cudaq` with admin rights to confirm installation steps.
RUN adduser --disabled-password --gecos '' cudaq && adduser cudaq sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN chown -R cudaq /home/cudaq && chgrp -R cudaq /home/cudaq
USER cudaq
WORKDIR /home/cudaq

## [Install]
ARG cuda_quantum_installer='install_cuda_quantum.*'
ADD "${cuda_quantum_installer}" install_cuda_quantum.sh
RUN echo "Installing CUDA Quantum..." \
    ## [>CUDAQuantumInstall]
    MPI_PATH=/usr/local/openmpi \
    sudo -E bash install_cuda_quantum.* --accept && \
    source /etc/profile
    ## [<CUDAQuantumInstall]

## [ADD tools for validation]
ADD scripts/validate_container.sh /home/cudaq/validate.sh
ADD docs/sphinx/examples/cpp /home/cudaq/examples

