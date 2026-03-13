# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This is a Dockerfile for building a container image that includes the CUDA-Q Realtime installation. 
# It uses a base image from NVIDIA's DOCA repository, which is required for running CUDA-Q Realtime applications.

ARG DOCA_VERSION=invalid
ARG CUDA_VERSION=invalid
FROM nvcr.io/nvidia/doca/doca:${DOCA_VERSION}-full-rt-cuda${CUDA_VERSION}.0.0

ARG CUDAQ_REALTIME_DIR=/opt/nvidia/cudaq/realtime

ADD . ${CUDAQ_REALTIME_DIR}

# Set LD_LIBRARY_PATH to include the CUDA-Q Realtime library path
ENV LD_LIBRARY_PATH="${CUDAQ_REALTIME_DIR}/lib:$LD_LIBRARY_PATH"
