# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file is used to build CUDA-Q NVQC service container to be deployed to NVCF.
#
# Usage:
# Must be built from the repo root with:
#   DOCKER_BUILDKIT=1 docker build -f docker/release/cudaq.nvqc.Dockerfile . --output out

# Base image is CUDA-Q image 
ARG base_image=nvcr.io/nvidia/nightly/cuda-quantum:latest
FROM $base_image as nvcf_image

# Run the tar command and then uncomment ADD cudaq.tar.gz ... in order to
# override the installation.
# tar czvf /workspaces/cuda-quantum/cudaq.tar.gz -C /usr/local/cudaq .
# ADD cudaq.tar.gz /opt/nvidia/cudaq

RUN sudo mkdir /nvqc_scripts
ADD tools/cudaq-qpud/nvqc_proxy.py /nvqc_scripts
ADD tools/cudaq-qpud/json_request_runner.py /nvqc_scripts
ADD scripts/nvqc_launch.sh /nvqc_scripts

ENTRYPOINT ["bash", "-l", "/nvqc_scripts/nvqc_launch.sh"]
