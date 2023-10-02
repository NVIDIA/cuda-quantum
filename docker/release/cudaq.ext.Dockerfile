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
RUN apt-get install -y --no-install-recommends \
        cuda-nvtx-11-8 libcusolver-11-8 libopenblas-openmp-dev \
        # just here for convenience:
        curl jq

# Make sure that apt-get remains updated at the end!;
# If we don't do that, then apt-get will get confused when some CUDA
# components are already installed but not all of them.

USER cudaq
