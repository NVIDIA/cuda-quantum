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
COPY "$assets" "$CUDA_QUANTUM_PATH/assets"

ADD ./scripts/migrate_assets.sh "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"
RUN for folder in `find "$CUDA_QUANTUM_PATH/assets"/* -maxdepth 0 -type d`; \
    do bash "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh" "$folder" && rm -rf "$folder"; done \
    && rm "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"

USER cudaq