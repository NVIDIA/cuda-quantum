#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
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
CUDA_DISTRIBUTION=${CUDA_DISTRIBUTION:-'rhel9'} # not really a great option, but allows some basic testing
CUDA_VERSION_SUFFIX=$(echo ${CUDART_VERSION:-'11.8'} | tr . -)
CUDA_PACKAGES=$(echo "cuda-cudart cuda-nvtx libcusolver libcublas" | sed "s/[^ ]*/&-${CUDA_VERSION_SUFFIX} /g")
CUDA_ARCH_FOLDER=$([ "$(uname -m)" == "aarch64" ] && echo sbsa || echo x86_64)

if [ "$pkg_manager" == "apt-get" ]; then
    ## [Prerequisites]
    apt-get update && apt-get install -y --no-install-recommends \
        sudo wget ca-certificates
    echo "apt-get install -y --no-install-recommends openssh-client" > install_sshclient.sh

    ## [C++ standard library]
    apt-get install -y --no-install-recommends ${LIBSTDCPP_PACKAGE:-'libstdc++-11-dev'}

    ## [CUDA runtime libraries]
    wget "${CUDA_DOWNLOAD_URL}/${CUDA_DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-keyring_1.1-1_all.deb"
    dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update 
    apt-get install -y --no-install-recommends ${CUDA_PACKAGES}
    rm cuda-keyring_1.1-1_all.deb

elif [ "$pkg_manager" == "dnf" ]; then
    ## [Prerequisites]
    dnf install -y --nobest --setopt=install_weak_deps=False \
        sudo 'dnf-command(config-manager)'
    echo "dnf install -y --nobest --setopt=install_weak_deps=False openssh-clients" > install_sshclient.sh

    ## [C++ standard library]
    LIBSTDCPP_PACKAGE=${LIBSTDCPP_PACKAGE:-'gcc-c++'}
    GCC_VERSION=`echo $LIBSTDCPP_PACKAGE | (egrep -o '[0-9]+' || true)`
    dnf install -y --nobest --setopt=install_weak_deps=False ${LIBSTDCPP_PACKAGE}
    enable_script=`find / -path '*gcc*' -path '*'$GCC_VERSIONS'*' -name enable`
    if [ -n "$enable_script" ]; then
        . "$enable_script"
    fi

    ## [CUDA runtime libraries]
    dnf config-manager --add-repo "${CUDA_DOWNLOAD_URL}/${CUDA_DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${CUDA_DISTRIBUTION}.repo"
    dnf install -y --nobest --setopt=install_weak_deps=False ${CUDA_PACKAGES}

elif [ "$pkg_manager" == "zypper" ]; then
    ## [Prerequisites]
    zypper clean --all && zypper --non-interactive up --no-recommends
    zypper --non-interactive in --no-recommends sudo gzip tar
    echo "zypper --non-interactive in --no-recommends openssh-clients" > install_sshclient.sh

    ## [C++ standard library]
    zypper --non-interactive in --no-recommends ${LIBSTDCPP_PACKAGE:-'gcc13-c++'}

    ## [CUDA runtime libraries]
    zypper ar "${CUDA_DOWNLOAD_URL}/${CUDA_DISTRIBUTION}/${CUDA_ARCH_FOLDER}/cuda-${CUDA_DISTRIBUTION}.repo"
    zypper --non-interactive --gpg-auto-import-keys in --no-recommends ${CUDA_PACKAGES}

else
    echo "Installation via $pkg_manager is not yet implemented." >&2
    (return 0 2>/dev/null) && return 1 || exit 1
fi

trap - ERR
(return 0 2>/dev/null) && return 0 || exit 0
