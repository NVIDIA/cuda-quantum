# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Single-shot build for the two CUDA-Q metapackage debs:
#
#   cudaq_${version}_all.deb           -> Depends on cuda-quantum-cu13 | cuda-quantum-cu12
#   cudaq-realtime_${version}_all.deb  -> Depends on cuda-quantum-realtime-cu13 | ...-cu12
#
# Neither file has a payload; they exist only to pull in a CUDA variant.
# Architecture-independent, so this runs once per release (not per matrix
# entry).
#
# Usage:
#   DOCKER_BUILDKIT=1 docker build \
#     -f docker/release/metapackage_deb.Dockerfile \
#     --build-arg cudaq_version=0.14.0 \
#     . --output out

FROM ubuntu:24.04 AS build
ARG DEBIAN_FRONTEND=noninteractive
ARG cudaq_version=0.0.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends dpkg-dev && \
    rm -rf /var/lib/apt/lists/*

COPY scripts/build_deb_metapackage.sh /usr/local/bin/build_deb_metapackage.sh
RUN chmod +x /usr/local/bin/build_deb_metapackage.sh && \
    mkdir -p /output && \
    bash /usr/local/bin/build_deb_metapackage.sh -f core     -V ${cudaq_version} -o /output -v && \
    bash /usr/local/bin/build_deb_metapackage.sh -f realtime -V ${cudaq_version} -o /output -v

FROM scratch
COPY --from=build /output/*.deb .
