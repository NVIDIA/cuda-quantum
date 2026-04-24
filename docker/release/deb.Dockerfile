# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file builds a Debian package (.deb) containing the CUDA-Q toolkit that
# can be installed on Debian/Ubuntu hosts with `apt install ./<file>.deb`.
# A suitable base image can be obtained by building docker/build/assets.Dockerfile.
# Sibling to docker/release/installer.Dockerfile (which produces the makeself
# self-extracting installer from the same staged tree).
#
# Usage:
# Must be built from the repo root with:
#   DOCKER_BUILDKIT=1 docker build -f docker/release/deb.Dockerfile . --output out

ARG base_image=ghcr.io/nvidia/cuda-quantum-assets:amd64-cu12-llvm-main
ARG additional_components=none
ARG cudaq_version=0.0.0

FROM $base_image AS additional_components_none
RUN echo "No additional components included."
FROM $base_image AS additional_components_assets
COPY assets /assets/
RUN source /cuda-quantum/scripts/configure_build.sh && \
    for folder in `find /assets/*$(uname -m)/* -maxdepth 0 -type d`; \
    do bash /cuda-quantum/scripts/migrate_assets.sh -s "$folder" && rm -rf "$folder"; \
    done

FROM additional_components_${additional_components} AS assets
ARG cudaq_version

# dpkg-deb + fakeroot + xz for building a .deb on AlmaLinux 8. lintian is a
# nice-to-have but may be missing on some base images; fall back cleanly.
# pigz + makeself are required because we reuse scripts/build_installer.sh
# for the staging work, and that script ends with a makeself --pigz archive
# step before we package the same tree as a deb.
RUN dnf install -y --nobest --setopt=install_weak_deps=False epel-release && \
    (dnf install -y --nobest --setopt=install_weak_deps=False \
        dpkg dpkg-dev fakeroot xz pigz lintian \
     || dnf install -y --nobest --setopt=install_weak_deps=False \
        dpkg dpkg-dev fakeroot xz pigz) && \
    git clone --filter=tree:0 https://github.com/megastep/makeself /makeself && \
    cd /makeself && git checkout release-2.5.0 && \
    ln -s /makeself/makeself.sh /usr/local/bin/makeself && \
    ln -s /makeself/makeself-header.sh /usr/local/bin/makeself-header.sh

# Stage the install tree via build_installer.sh (its -o output is unused by
# this Dockerfile; we just want the side-effect of populating
# build/cuda_quantum_assets/cudaq/). Then pack it as a .deb.
RUN cd /cuda-quantum && \
    bash scripts/build_installer.sh \
        -d \
        -c $(echo ${CUDA_VERSION} | cut -d . -f1) \
        -o /tmp/out-stage \
        -v && \
    bash scripts/build_deb.sh \
        -d \
        -f core \
        -c $(echo ${CUDA_VERSION} | cut -d . -f1) \
        -i /cuda-quantum/build/cuda_quantum_assets/cudaq \
        -V ${cudaq_version} \
        -o /output \
        -v

FROM scratch
COPY --from=assets /output/*.deb .
