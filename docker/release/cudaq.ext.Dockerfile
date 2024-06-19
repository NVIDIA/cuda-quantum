# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=nvcr.io/nvidia/nightly/cuda-quantum:latest-base
FROM $base_image

USER root

# Copy over additional CUDA-Q assets.
ARG assets=./assets
COPY "$assets" "$CUDA_QUANTUM_PATH/assets/"
ADD ./scripts/migrate_assets.sh "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"
# Remove the base build_info.txt because the migration intentionally does not overwrite
# existing files, but adds its own entries to the build info.
RUN rm "$CUDA_QUANTUM_PATH/build_info.txt"
RUN if [ -d "$CUDA_QUANTUM_PATH/assets/documentation" ]; then \
        rm -rf "$CUDA_QUANTUM_PATH/docs" && mkdir -p "$CUDA_QUANTUM_PATH/docs"; \
        mv "$CUDA_QUANTUM_PATH/assets/documentation"/* "$CUDA_QUANTUM_PATH/docs"; \
        rmdir "$CUDA_QUANTUM_PATH/assets/documentation"; \
    fi && \
    for folder in `find "$CUDA_QUANTUM_PATH/assets/$(uname -m)"/* -maxdepth 0 -type d -not -name cudaq`; \
    do bash "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh" -s "$folder" && rm -rf "$folder"; done \
    && bash "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh" -s "$CUDA_QUANTUM_PATH/assets/$(uname -m)" \
    && rm -rf "$CUDA_QUANTUM_PATH/assets" "$CUDA_QUANTUM_PATH/bin/migrate_assets.sh"

# Install additional runtime dependencies.
RUN apt-get install -y --no-install-recommends \
        libcusolver-11-8 libcublas-11-8 cuda-cudart-11-8 \
        # just here for convenience:
        curl jq 
RUN if [ -x "$(command -v pip)" ]; then \
        apt-get install -y --no-install-recommends gcc libpython3-dev \
        && pip install --no-cache-dir jupyterlab; \
        if [ -n "$MPI_ROOT" ]; then \
            pip install --no-cache-dir mpi4py~=3.1; \
        fi; \
    fi
# Make sure that apt-get remains updated at the end!;
# If we don't do that, then apt-get will get confused when some CUDA
# components are already installed but not all of them.

# Include helper scripts and configurations to facilitate 
# development with VS Code and JupterLab.
# See also https://github.com/microsoft/vscode/issues/60#issuecomment-161792005
ARG vscode_config=.vscode
COPY "${vscode_config}" /home/cudaq/.vscode
RUN echo -e '#! /bin/bash \n\
    if [ ! -x "$(command -v code)" ]; then \n\
        os=$([ "$(uname -m)" == "aarch64" ] && echo cli-alpine-arm64 || echo cli-alpine-x64) \n\
        curl -Lk "https://code.visualstudio.com/sha/download?build=stable&os=$os" --output vscode_cli.tar.gz \n\
        tar -xf vscode_cli.tar.gz && rm vscode_cli.tar.gz && sudo mv code /usr/bin/ \n\
    fi \n\
    code "$@"' > "$CUDA_QUANTUM_PATH/bin/vscode-setup" \
    && chmod +x "$CUDA_QUANTUM_PATH/bin/vscode-setup"
RUN echo -e '#! /bin/bash \n\
    jupyter-lab --no-browser --ip=* --ServerApp.allow_origin=* --IdentityProvider.token="$@" \n\
    ' > "$CUDA_QUANTUM_PATH/bin/jupyter-lab-setup" \
    && chmod +x "$CUDA_QUANTUM_PATH/bin/jupyter-lab-setup"

RUN chown -R cudaq /home/cudaq && chgrp -R cudaq /home/cudaq
USER cudaq
