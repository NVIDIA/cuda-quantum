# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=redhat/ubi8:8.0
ARG base_image_mpibuild=amd64/almalinux:8

# [OpenMPI Installation]
FROM ${base_image_mpibuild} as mpibuild
ARG base_image_mpibuild
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

## [Prerequisites]
ADD docker/test/installer/dependencies.sh /runtime_dependencies.sh
RUN bash runtime_dependencies.sh ${base_image_mpibuild}
RUN dnf install -y --nobest --setopt=install_weak_deps=False \
        autoconf libtool flex make wget \
        gcc-toolset-11 cuda-cudart-devel-11-8

## [Build]
ADD scripts/configure_build.sh /cuda-quantum/scripts/configure_build.sh
RUN source /cuda-quantum/scripts/configure_build.sh build-openmpi

# [CUDA Quantum Installation]
FROM ${base_image}
ARG base_image

ARG DEBIAN_FRONTEND=noninteractive
ADD docker/test/installer/dependencies.sh /runtime_dependencies.sh
RUN bash runtime_dependencies.sh ${base_image}

## [MPI Installation]
ADD docker/test/installer/mpi_cuda_check.cpp mpi_cuda_check.cpp
COPY --from=mpibuild /usr/local/openmpi/ /usr/local/openmpi
RUN ln -s /usr/local/openmpi/bin/mpiexec /bin/mpiexec && \
    /usr/local/openmpi/bin/mpic++ mpi_cuda_check.cpp -o check.x && \
    mpiexec -np 1 ./check.x

# Create new user `cudaq` with admin rights to confirm installation steps.
RUN useradd cudaq && mkdir -p /etc/sudoers.d && \
    echo 'cudaq ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers.d/020_cudaq
RUN mkdir -p /home/cudaq && chown -R cudaq /home/cudaq && chgrp -R cudaq /home/cudaq
USER cudaq
WORKDIR /home/cudaq

## [Install]
ARG cuda_quantum_installer='install_cuda_quantum.*'
ADD "${cuda_quantum_installer}" install_cuda_quantum.sh
RUN echo "Installing CUDA Quantum..." && \
    ## [>CUDAQuantumInstall]
    MPI_PATH=/usr/local/openmpi \
    sudo -E bash install_cuda_quantum.* --accept && . /etc/profile
    ## [<CUDAQuantumInstall]
RUN . /etc/profile && nvq++ --help

## [ADD tools for validation]
ADD scripts/validate_container.sh /home/cudaq/validate.sh
ADD docs/sphinx/examples/cpp /home/cudaq/examples
ENTRYPOINT ["bash", "-l"]

