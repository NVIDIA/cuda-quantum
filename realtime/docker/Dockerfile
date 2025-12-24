# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=ghcr.io/nvidia/cuda-quantum-dev:ext-cu13.0-gcc11-main
FROM $base_image

ARG TARGETARCH
ARG DOCA_VERSION=3.1.0
ARG CACHEBUST=1
ARG DEBIAN_FRONTEND=noninteractive

RUN echo "Target architecture is $TARGETARCH"

WORKDIR /opt

RUN apt-get update && apt-get install -y --no-install-recommends \
        gpg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Configure the DOCA APT repository
RUN if [ "${TARGETARCH}" = "amd64" ]; then \
        DOCA_ARCH="x86_64"; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        DOCA_ARCH="arm64-sbsa"; \
    else \
        echo "Unknown architecture: $TARGETARCH"; \
        exit 1; \
    fi \
    && DISTRO=$(. /etc/os-release && echo ${ID}${VERSION_ID}) \
    && DOCA_REPO_LINK=https://linux.mellanox.com/public/repo/doca/${DOCA_VERSION}/${DISTRO}/${DOCA_ARCH} \
    && echo "Using DOCA_REPO_LINK=${DOCA_REPO_LINK}" \
    && LOCAL_GPG_KEY_PATH="/usr/share/keyrings/mellanox-archive-keyring.gpg" \
    && curl -fsSL ${DOCA_REPO_LINK}/GPG-KEY-Mellanox.pub | gpg --dearmor | tee ${LOCAL_GPG_KEY_PATH} > /dev/null \
    && echo "deb [signed-by=${LOCAL_GPG_KEY_PATH}] ${DOCA_REPO_LINK} ./" | tee /etc/apt/sources.list.d/mellanox.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkgconf \
        mlnx-dpdk-dev \
        doca-sdk \
        libdoca-sdk-gpunetio-dev \
        libdoca-sdk-verbs-dev \
        mlnx-ofed-kernel-utils \
        ibverbs-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Soft-RoCE (rxe) support packages
# Note: To use Soft-RoCE in the container, run with:
#   --cap-add=NET_ADMIN --cap-add=SYS_MODULE --device=/dev/infiniband
# And ensure rdma_rxe kernel module is loaded on the host:
#   modprobe rdma_rxe
#   rdma link add rxe0 type rxe netdev <interface>
RUN apt-get update && apt-get install -y --no-install-recommends \
        rdma-core \
        libibverbs-dev \
        librdmacm-dev \
        iproute2 \
        kmod \
        perftest \
        infiniband-diags \
    && rm -rf /var/lib/apt/lists/*
