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

USER root

ENV CUDAQ_LOG_LEVEL=info
ADD ./scripts/launch_qpud.sh /launch_qpud.sh
RUN chmod +x /launch_qpud.sh

# Start the cudaq-qpud service
ENTRYPOINT ["/launch_qpud.sh"]
