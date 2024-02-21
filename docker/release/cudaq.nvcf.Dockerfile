# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file is used to build CUDA Quantum NVCF service container.
#
# Usage:
# Must be built from the repo root with:
#   DOCKER_BUILDKIT=1 docker build -f docker/release/cudaq.nvcf.Dockerfile . --output out

# Base image is CUDA Quantum image 
ARG base_image=nvcr.io/nvidia/nightly/cuda-quantum:latest
FROM $base_image as nvcf_image

ENV CUDAQ_LOG_LEVEL=info

# Launch script: launch cudaq-qpud (nvcf mode) with MPI ranks == Number of NVIDIA GPUs
RUN echo 'mpiexec -np $(nvidia-smi --list-gpus | wc -l) cudaq-qpud --type nvcf' > launch.sh

# Start the cudaq-qpud service
ENTRYPOINT ["bash", "-l", "launch.sh"]
