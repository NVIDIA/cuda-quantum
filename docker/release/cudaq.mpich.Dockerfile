# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file is used to build CUDA-Q image with MPICH MPI rather than OpenMPI.
# IMPORTANT NOTE: this is intended to create an MPICH-compatible image,
# whereby mpi4py and CUDA-Q MPI plugin are linked against MPICH's '.so'.
# This makes it compatible with MPI hotswap/injection scenario, e.g., HPC centers.
# The MPICH installation in this image is thus a barebone/vanilla one (e.g., no CUDA-aware capability), 
# and may not be suitable for standalone usage.
# Always use the CUDA-Q image (with full OpenMPI installation) in that case.

#
# Usage:
# Must be built from the repo root with:
#   DOCKER_BUILDKIT=1 docker build -f docker/release/cudaq.mpich.Dockerfile . --output out

# Base image is CUDA-Q image 
ARG base_image=nvcr.io/nvidia/nightly/cuda-quantum:latest
FROM $base_image 

USER root

# Remove OpenMPI
RUN rm -rf /usr/local/gdrcopy /usr/local/munge /usr/local/openmpi /usr/local/pmi /usr/local/ucx /usr/local/pmix
ENV MPI_HOME=
ENV MPI_ROOT=

# Install MPICH
WORKDIR /tmp
RUN \
    apt-get update        && \
    apt-get install --yes    \
        build-essential      \
        gfortran          && \
    apt-get clean all

ARG mpich=4.1.1
ARG mpich_prefix=mpich-$mpich
ENV MPI_PATH=/usr/local/mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure -prefix=$MPI_PATH                                           && \
    make -j$(nproc)                                                         && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix                                                    && \
    /sbin/ldconfig

ENV PATH="$MPI_PATH/bin:$PATH"

# Reinstall mpi4py
RUN python3 -m pip uninstall -y mpi4py && MPICC=$(which mpicc) python3 -m pip install -v --no-binary mpi4py mpi4py

# Rebuild CUDA-Q MPI plugin
RUN rm $CUDA_QUANTUM_PATH/lib/plugins/libcudaq-comm-plugin.so && bash $CUDA_QUANTUM_PATH/distributed_interfaces/activate_custom_mpi.sh

# Switch back to cudaq user
USER cudaq
WORKDIR /home/cudaq