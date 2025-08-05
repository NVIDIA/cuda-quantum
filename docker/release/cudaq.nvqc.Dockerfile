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
# Flag to control whether to install/clone prerequisites
ARG WITH_TPLS=false

FROM $base_image AS nvcf_image

# Copy and run the install_prerequisites script into the image
RUN if [ "${WITH_TPLS}" = "true" ]; then \
        echo "WITH_TPLS=true; adding install_prerequisites.sh"; \
        mkdir -p /tmp && \
        cp scripts/install_prerequisites.sh /tmp/install_prerequisites.sh && \
        chmod +x /tmp/install_prerequisites.sh && \
        /tmp/install_prerequisites.sh -m -k; \
    else \
        echo "Default build without prerequisites;" \
    fi

# run it in "keep-sources" mode but skip any builds (-m), since base image already has them
RUN /tmp/install_prerequisites.sh -m -k

# Run the tar command and then uncomment ADD cudaq.tar.gz ... in order to
# override the installation.
# tar czvf /workspaces/cuda-quantum/cudaq.tar.gz -C /usr/local/cudaq .
# ADD cudaq.tar.gz /opt/nvidia/cudaq

RUN sudo mkdir /nvqc_scripts
ADD tools/cudaq-qpud/nvqc_proxy.py /nvqc_scripts
ADD tools/cudaq-qpud/json_request_runner.py /nvqc_scripts
ADD scripts/nvqc_launch.sh /nvqc_scripts

ENTRYPOINT ["bash", "-l", "/nvqc_scripts/nvqc_launch.sh"]
