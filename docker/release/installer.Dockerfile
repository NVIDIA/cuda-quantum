# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds a self-extractable CUDA-Q archive that can be installed
# on a compatible Linux host system; see also https://makeself.io/.
# A suitable base image can be obtained by building docker/build/assets.Dockerfile.
#
# Usage:
# Must be built from the repo root with:
#   DOCKER_BUILDKIT=1 docker build -f docker/release/installer.Dockerfile . --output out

ARG base_image=ghcr.io/nvidia/cuda-quantum-assets:amd64-cu12-llvm-main
ARG additional_components=none

FROM $base_image AS additional_components_none
RUN echo "No additional components included."
FROM $base_image AS additional_components_assets
COPY assets /assets/
RUN source /cuda-quantum/scripts/configure_build.sh && \
    for folder in `find /assets/*$(uname -m)/* -maxdepth 0 -type d`; \
    do bash /cuda-quantum/scripts/migrate_assets.sh -s "$folder" && rm -rf "$folder"; \
    done

# [Installer]
FROM additional_components_${additional_components} AS assets

# Install makeself
RUN git clone --filter=tree:0 https://github.com/megastep/makeself /makeself && \
    cd /makeself && git checkout release-2.5.0 && \
    ln -s /makeself/makeself.sh /usr/local/bin/makeself && \
    ln -s /makeself/makeself-header.sh /usr/local/bin/makeself-header.sh

# Build installer using unified script
# -d: Docker mode (uses configure_build.sh paths, skips verification)
# -c: CUDA variant extracted from CUDA_VERSION env var
# -o: Output directory for the installer
RUN cd /cuda-quantum && \
    bash scripts/build_installer.sh \
        -d \
        -c $(echo ${CUDA_VERSION} | cut -d . -f1) \
        -o /output

FROM scratch
COPY --from=assets /output/install_cuda_quantum* .
COPY --from=assets /cuda-quantum/wheelhouse/* . 
