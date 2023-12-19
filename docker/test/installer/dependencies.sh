#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

case $1 in
    *ubuntu*)
        pkg_manager=apt-get
        distro=ubuntu2204
    ;;
    *debian*) 
        pkg_manager=apt-get
        distro=debian12
    ;;
    *almalinux*|*redhat*|*fedora*)
        pkg_manager=dnf
        distro=rhel9
    ;;
    *opensuse*) 
        pkg_manager=zypper
        distro=sles15
    ;;
    *) echo "No package manager configured for $1" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
    ;;
esac

CUDA_DOWNLOAD_URL=https://developer.download.nvidia.com/compute/cuda/repos
CUDA_PACKAGES="cuda-cudart-11-8 cuda-nvtx-11-8 libcusolver-11-8 libcublas-11-8"
CUDA_ARCH_FOLDER=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64)

if [ "$pkg_manager" == "apt-get" ]; then
    ## [Prerequisites]
    apt-get update && apt-get install -y --no-install-recommends wget ca-certificates

    ## [C++ standard library]
    apt-get install -y --no-install-recommends libstdc++-11-dev

    ## [CUDA runtime libraries]
    wget "${CUDA_DOWNLOAD_URL}/${distro}/${CUDA_ARCH_FOLDER}/cuda-keyring_1.0-1_all.deb"
    dpkg -i cuda-keyring_1.0-1_all.deb && apt-get update 
    apt-get install -y --no-install-recommends ${CUDA_PACKAGES}
    rm cuda-keyring_1.0-1_all.deb

elif [ "$pkg_manager" == "dnf" ]; then
    ## [Prerequisites]
    dnf install -y --nobest --setopt=install_weak_deps=False \
        'dnf-command(config-manager)'

    ## [C++ standard library]
    dnf 

    ## [CUDA runtime libraries]
    dnf config-manager --add-repo "${CUDA_DOWNLOAD_URL}/${distro}/${CUDA_ARCH_FOLDER}/cuda-${distro}.repo"
    dnf install -y --nobest --setopt=install_weak_deps=False ${CUDA_PACKAGES}

else
    echo "Installation via $pkg_manager is not yet implemented." >&2
    (return 0 2>/dev/null) && return 1 || exit 1
fi
