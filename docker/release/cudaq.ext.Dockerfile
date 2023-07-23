# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=ghcr.io/nvidia/cuda-quantum:latest
FROM $base_image

USER root

ARG assets=./assets
COPY "$assets" "$CUDA_QUANTUM_PATH/assets/"

ADD ./scripts/migrate_assets.sh "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"
RUN for folder in `find "$CUDA_QUANTUM_PATH/assets"/* -maxdepth 0 -type d`; \
    do bash "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh" "$folder" && rm -rf "$folder"; done \
    && rm "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"

# Install additional runtime dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvtx-11-8 libcusolver-dev-11-8 libopenblas-openmp-dev \
        # just here for convenience:
        curl jq \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

USER cudaq
