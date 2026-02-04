# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This Dockerfile is used to store macOS build artifacts in GHCR.
# Since Docker cannot build native macOS images, we use an Ubuntu container
# as a storage mechanism for macOS artifacts built on bare metal runners.
#
# The artifacts are packaged into a tar.gz file and stored in this container,
# then extracted on macOS runners when needed.
#
# Usage:
#   1. Build prerequisites on macOS runner
#   2. Package: tar -czf artifacts.tar.gz -C $HOME .local .llvm-project
#   3. Build this image with the artifacts
#   4. Push to GHCR
#   5. Pull and extract on consuming jobs

FROM ubuntu:24.04

LABEL org.opencontainers.image.title="cuda-quantum-macos-cache"
LABEL org.opencontainers.image.description="macOS build artifact cache for CUDA Quantum"

# Copy the artifacts tar file into the container
COPY artifacts.tar.gz /macos-artifacts.tar.gz
