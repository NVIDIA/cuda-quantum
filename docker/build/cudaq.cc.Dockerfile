# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:compiler-rt
FROM $base_image

ENV CUDAQ_REPO_ROOT=/workspaces/cuda-quantum
ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq
ENV PATH="$CUDAQ_INSTALL_PREFIX/bin:${PATH}"
ENV PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH}"

ARG workspace=.
ARG destination="$CUDAQ_REPO_ROOT"
ADD "$workspace" "$destination"

WORKDIR "$destination"

# build cudaq with cc module
ARG install=
RUN if [ -n "$install" ]; \
    then \
        expected_prefix=$CUDAQ_INSTALL_PREFIX; \
        install=`echo $install | xargs` && export $install; \
        bash scripts/build_cudaq.sh; \
        if [ ! "$?" -eq "0" ]; then \
            exit 1; \
        elif [ "$CUDAQ_INSTALL_PREFIX" != "$expected_prefix" ]; then \
            mkdir -p "$expected_prefix"; \
            mv "$CUDAQ_INSTALL_PREFIX"/* "$expected_prefix"; \
            rmdir "$CUDAQ_INSTALL_PREFIX"; \
        fi \
    fi
