# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=ghcr.io/nvidia/cuda-quantum:latest-base
FROM $base_image

USER root

ARG assets=./assets
COPY "$assets" "$CUDA_QUANTUM_PATH/assets/"

ADD ./scripts/migrate_assets.sh "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"
RUN for folder in `find "$CUDA_QUANTUM_PATH/assets"/*$(uname -m)/* -maxdepth 0 -type d`; \
    do bash "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh" "$folder" && rm -rf "$folder"; done \
    && rm "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"

# Install additional runtime dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas-openmp-dev \
        # just here for convenience:
        curl jq \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Run apt-get update to ensure that apt-get knows about CUDA packages
# if the base image configures has added the CUDA keyring.
# If we don't do that, then apt-get will get confused when some CUDA
# components are already installed but not all of it.
RUN apt-get update

USER cudaq
