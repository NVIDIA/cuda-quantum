# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This Dockerfile builds a container to be deployed to NVCF for the CUDA-Q NVQC service,
# extending the official CUDA-Q image and adding third-party library sources.

# Usage:
#   docker build -f docker/release/cudaq.nvqc.all.Dockerfile .

# Base image is CUDA-Q image 
ARG base_image=nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
FROM $base_image

USER root

# Ensure base directories exists
RUN mkdir -p /opt/tpls /opt/tpls/customizations /opt/tpls/json

# Copy in third-party repositories
COPY tpls/ /opt/tpls/

# Copy the nvqc scripts required for the server
RUN sudo mkdir /nvqc_scripts
ADD tools/cudaq-qpud/nvqc_proxy.py /nvqc_scripts
ADD tools/cudaq-qpud/json_request_runner.py /nvqc_scripts
ADD scripts/nvqc_launch.sh /nvqc_scripts

# Set Permissions
RUN chown -R cudaq:cudaq /opt/tpls /nvqc_scripts

USER cudaq
WORKDIR /home/cudaq

ENTRYPOINT ["bash", "-l", "/nvqc_scripts/nvqc_launch.sh"]
