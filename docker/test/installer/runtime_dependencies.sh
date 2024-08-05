#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

trap '(return 0 2>/dev/null) && return 1 || exit 1' ERR

case $1 in
    *ubuntu*)
        pkg_manager=apt-get
    ;;
    *debian*) 
        pkg_manager=apt-get
    ;;
    *almalinux*|*redhat*)
        pkg_manager=dnf
    ;;
    *fedora*)
        pkg_manager=dnf
    ;;
    *opensuse*) 
        pkg_manager=zypper
    ;;
    *)  echo "No package manager configured for $1" >&2
        (return 0 2>/dev/null) && return 1 || exit 1
    ;;
esac

CUDA_DOWNLOAD_URL=https://developer.download.nvidia.com/compute/cuda/repos
CUDA_VERSION_SUFFIX=$(echo ${CUDART_VERSION:-'11.8'} | tr . -)
CUDA_PACKAGES=$(echo "cuda-cudart libcusolver libcublas" | sed "s/[^ ]*/&-${CUDA_VERSION_SUFFIX} /g")
CUDA_ARCH_FOLDER=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64)

if [ "$pkg_manager" == "apt-get" ]; then
    ## [Prerequisites]
    apt-get update && apt-get install -y --no-install-recommends \
        sudo wget ca-certificates
    echo "apt-get install -y --no-install-recommends openssh-client" > install_sshclient.sh

    ## [C development headers]
    if [ -n "${LIBCDEV_PACKAGE}" ]; then
        apt-get install -y --no-install-recommends ${LIBCDEV_PACKAGE}
    fi

    ## [CUDA runtime libraries]
    if [ -n "${CUDA_DISTRIBUTION}" ]; then
        wget "${CUDA_DOWNLOAD_URL}/${CUDA_DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-keyring_1.1-1_all.deb"
        dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update 
        apt-get install -y --no-install-recommends ${CUDA_PACKAGES}
        rm cuda-keyring_1.1-1_all.deb
    fi

elif [ "$pkg_manager" == "dnf" ]; then
    ## [Prerequisites]
    dnf install -y --nobest --setopt=install_weak_deps=False \
        sudo 'dnf-command(config-manager)'
    echo "dnf install -y --nobest --setopt=install_weak_deps=False openssh-clients" > install_sshclient.sh

    ## [C development headers]
    if [ -n "${LIBCDEV_PACKAGE}" ]; then
        dnf install -y --nobest --setopt=install_weak_deps=False ${LIBCDEV_PACKAGE}
    fi

    ## [CUDA runtime libraries]
    if [ -n "${CUDA_DISTRIBUTION}" ]; then
        dnf config-manager --add-repo "${CUDA_DOWNLOAD_URL}/${CUDA_DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${CUDA_DISTRIBUTION}.repo"
        dnf install -y --nobest --setopt=install_weak_deps=False ${CUDA_PACKAGES}
    fi

elif [ "$pkg_manager" == "zypper" ]; then
    ## [Prerequisites]
    zypper clean --all && zypper ref && zypper --non-interactive up --no-recommends
    zypper --non-interactive in --no-recommends sudo gzip tar
    echo "zypper --non-interactive in --no-recommends openssh-clients" > install_sshclient.sh

    ## [C development headers]
    if [ -n "${LIBCDEV_PACKAGE}" ]; then
        zypper --non-interactive in --no-recommends ${LIBCDEV_PACKAGE}
    fi

    ## [CUDA runtime libraries]
    if [ -n "${CUDA_DISTRIBUTION}" ]; then
        zypper ar "${CUDA_DOWNLOAD_URL}/${CUDA_DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${CUDA_DISTRIBUTION}.repo"
        zypper --non-interactive --gpg-auto-import-keys in --no-recommends ${CUDA_PACKAGES}
    fi

else
    echo "Installation via $pkg_manager is not yet implemented." >&2
    (return 0 2>/dev/null) && return 1 || exit 1
fi

trap - ERR
(return 0 2>/dev/null) && return 0 || exit 0
