# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=nvcr.io/nvidia/nightly/cuda-quantum:latest-base
FROM $base_image

USER root

# Copy over additional CUDA Quantum assets.
ARG assets=./assets
COPY "$assets" "$CUDA_QUANTUM_PATH/assets/"
ADD ./scripts/migrate_assets.sh "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"
RUN if [ -d "$CUDA_QUANTUM_PATH/assets/documentation" ]; then \
        rm -rf "$CUDA_QUANTUM_PATH/docs" && mkdir -p "$CUDA_QUANTUM_PATH/docs"; \
        mv "$CUDA_QUANTUM_PATH/assets/documentation"/* "$CUDA_QUANTUM_PATH/docs"; \
        rmdir "$CUDA_QUANTUM_PATH/assets/documentation"; \
    fi && \
    for folder in `find "$CUDA_QUANTUM_PATH/assets"/*$(uname -m)/* -maxdepth 0 -type d`; \
    do bash "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh" "$folder" && rm -rf "$folder"; done \
    && rm "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"

# Install additional runtime dependencies.
RUN apt-get install -y --no-install-recommends \
        cuda-nvtx-11-8 libcusolver-11-8 libopenblas-openmp-dev \
        # just here for convenience:
        curl jq 
RUN if [ -x "$(command -v pip)" ]; then \
        apt-get install -y --no-install-recommends gcc \
        && pip install --no-cache-dir jupyterlab matplotlib; \
        if [ -n "$MPI_ROOT" ]; then \
            pip install --no-cache-dir mpi4py~=3.1; \
        fi; \
    fi
# Make sure that apt-get remains updated at the end!;
# If we don't do that, then apt-get will get confused when some CUDA
# components are already installed but not all of them.

# Include VS Code configurations to facilitate development.
# It would be nice to include the CLI to enable tunneling into the container by default, 
# but we should build it from source then, see also
# https://github.com/microsoft/vscode/issues/60#issuecomment-161792005
ARG vscode_config=.vscode
COPY "${vscode_config}" /home/cudaq/.vscode

RUN chown -R cudaq /home/cudaq && chgrp -R cudaq /home/cudaq
USER cudaq
