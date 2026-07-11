#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

trap '(return 0 2>/dev/null) && return 1 || exit 1' ERR

# Tolerate transient apt/dnf mirror failures.
if [ -d /etc/apt/apt.conf.d ]; then
    echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries 2>/dev/null || \
        sudo sh -c 'echo "Acquire::Retries \"5\";" > /etc/apt/apt.conf.d/80-retries' 2>/dev/null || true
fi

# Retry dnf operations to tolerate transient CDN/mirror races. NVIDIA's CUDA
# repo rotates its repodata, and if the hashed *-primary.xml.gz referenced by a
# just-fetched repomd.xml is replaced before dnf downloads it, the install dies
# with a 404 on metadata. Refreshing metadata between attempts re-fetches the
# current repomd so the retry sees consistent files.
dnf_retry() {
    for n in 1 2 3 4 5; do
        dnf "$@" && return 0
        [ $n -lt 5 ] && { dnf clean all >/dev/null 2>&1 || true; sleep $((5*n)); }
    done
    return 1
}

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
CUDA_ARCH_FOLDER=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64)
CUDA_VERSION_SUFFIX=$(echo ${CUDART_VERSION:-'12.0'} | tr . -)
CUDA_PACKAGES=$(echo "cuda-cudart cuda-nvrtc libcusolver libcublas libcurand libcusparse" | sed "s/[^ ]*/&-${CUDA_VERSION_SUFFIX} /g")
if [ $(echo ${CUDART_VERSION} | cut -d . -f1) -gt 11 ]; then 
    CUDA_PACKAGES+=" libnvjitlink-${CUDA_VERSION_SUFFIX}"
fi
if [ -n "${INSTALL_CMAKE_VERSION}" ]; then
    INSTALL_WGET=true
fi

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

    ## [Extra validation packages]
    if [ -n "${VALIDATION_PACKAGES}" ]; then
        apt-get install -y --no-install-recommends ${VALIDATION_PACKAGES}
    fi

    if [ -n "${INSTALL_WGET}" ]; then
        apt-get install -y --no-install-recommends wget
    fi

elif [ "$pkg_manager" == "dnf" ]; then
    ## [Prerequisites]
    dnf_retry install -y --nobest --setopt=install_weak_deps=False \
        sudo 'dnf-command(config-manager)'
    echo "dnf install -y --nobest --setopt=install_weak_deps=False openssh-clients" > install_sshclient.sh

    ## [C development headers]
    if [ -n "${LIBCDEV_PACKAGE}" ]; then
        dnf_retry install -y --nobest --setopt=install_weak_deps=False ${LIBCDEV_PACKAGE}
    fi

    ## [CUDA runtime libraries]
    if [ -n "${CUDA_DISTRIBUTION}" ]; then
        dnf config-manager --add-repo "${CUDA_DOWNLOAD_URL}/${CUDA_DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${CUDA_DISTRIBUTION}.repo"
        dnf_retry install -y --nobest --setopt=install_weak_deps=False ${CUDA_PACKAGES}
    fi

    ## [Extra validation packages]
    if [ -n "${VALIDATION_PACKAGES}" ]; then
        dnf_retry install -y --nobest --setopt=install_weak_deps=False ${VALIDATION_PACKAGES}
    fi

    if [ -n "${INSTALL_WGET}" ]; then
        dnf_retry install -y --nobest --setopt=install_weak_deps=False wget
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

    ## [Extra validation packages]
    if [ -n "${VALIDATION_PACKAGES}" ]; then
        zypper --non-interactive in --no-recommends ${VALIDATION_PACKAGES}
    fi

    if [ -n "${INSTALL_WGET}" ]; then
        zypper --non-interactive in --no-recommends wget
    fi

else
    echo "Installation via $pkg_manager is not yet implemented." >&2
    (return 0 2>/dev/null) && return 1 || exit 1
fi

## [CMake via Kitware installer]
if [ -n "${INSTALL_CMAKE_VERSION}" ]; then
    wget -q "https://github.com/Kitware/CMake/releases/download/v${INSTALL_CMAKE_VERSION}/cmake-${INSTALL_CMAKE_VERSION}-linux-$(uname -m).sh" \
        -O cmake-install.sh
    bash cmake-install.sh --skip-license --exclude-subdir --prefix=/usr/local
    rm cmake-install.sh
fi

trap - ERR
(return 0 2>/dev/null) && return 0 || exit 0
