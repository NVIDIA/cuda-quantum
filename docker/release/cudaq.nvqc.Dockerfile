# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file is used to build CUDA Quantum NVQC service container to be deployed to NVCF.
#
# Usage:
# Must be built from the repo root with:
#   DOCKER_BUILDKIT=1 docker build -f docker/release/cudaq.nvqc.Dockerfile . --output out

# Base image is CUDA Quantum image 
ARG base_image=nvcr.io/nvidia/nightly/cuda-quantum:latest
FROM $base_image as nvcf_image

ENV CUDAQ_LOG_LEVEL=info

# Launch script: launch cudaq-qpud (nvcf mode) with MPI ranks == Number of NVIDIA GPUs
# IMPORTANT: NVCF function must set container environment variable `NUM_GPUS` equal to the number of GPUs on the target platform.
# This will allow clients to query the function capability (number of GPUs) by looking at function info.
# The below entry point script helps prevent mis-configuration by checking that functions are created and deployed appropriately.
RUN echo 'if [[ "$NUM_GPUS" == "$(nvidia-smi --list-gpus | wc -l)" ]]; then mpiexec -np $(nvidia-smi --list-gpus | wc -l) cudaq-qpud --type nvcf; else echo "Invalid Deployment: Number of GPUs does not match the hardware" && exit 1; fi' > launch.sh

# Start the cudaq-qpud service
ENTRYPOINT ["bash", "-l", "launch.sh"]
