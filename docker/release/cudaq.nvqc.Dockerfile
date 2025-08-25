# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file is used to build CUDA-Q NVQC service container to be deployed to NVCF.
#
# Usage:
# Must be built from the repo root with:
#   # to skip prerequisites (default)
#   docker build -f docker/release/cudaq.nvqc.Dockerfile .
#
#   # to install and clone prerequisites
#   docker build --build-arg WITH_TPLS=true \
#       -f docker/release/cudaq.nvqc.Dockerfile .

# Base image is CUDA-Q image 
ARG base_image=nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
FROM $base_image AS nvcf_image

# Flag to control whether to install/clone prerequisites
ARG WITH_TPLS=false

# COPY install_prerequisites into the image
RUN sudo mkdir -p /tmp
COPY --chmod=0755 scripts/install_prerequisites.sh /tmp/install_prerequisites.sh
COPY .gitmodules /tmp/.gitmodules

# Use the following command to generate the tpls_commits.lock file which
# contains the pinned commits for the submodules in /tpls directory. This is
# needed to generate NVQC image with tpls source code.
# git config --file .gitmodules --get-regexp '^submodule\..*\.path$' \
#         | awk '{print $2}' \
#         | while read p; do printf "%s %s\n" "$(git rev-parse HEAD:$p)" "$p"; done \
#         > tpls_commits.lock

# Uncomment the following line to copy the tpls_commits.lock file into the image
# COPY tpls_commits.lock /tmp/tpls_commits.lock

# Copy and run the install_prerequisites script into the image
RUN if [ "${WITH_TPLS}" = "true" ]; then \
        echo "WITH_TPLS=true; adding install_prerequisites.sh"; \
        bash /tmp/install_prerequisites.sh -l /tmp/tpls_commits.lock; \
    else \
        echo "Default build without prerequisites";  \
    fi

# Run the tar command and then uncomment ADD cudaq.tar.gz ... in order to
# override the installation.
# tar czvf /workspaces/cuda-quantum/cudaq.tar.gz -C /usr/local/cudaq .
# ADD cudaq.tar.gz /opt/nvidia/cudaq

RUN sudo mkdir /nvqc_scripts
ADD tools/cudaq-qpud/nvqc_proxy.py /nvqc_scripts
ADD tools/cudaq-qpud/json_request_runner.py /nvqc_scripts
ADD scripts/nvqc_launch.sh /nvqc_scripts

ENTRYPOINT ["bash", "-l", "/nvqc_scripts/nvqc_launch.sh"]
