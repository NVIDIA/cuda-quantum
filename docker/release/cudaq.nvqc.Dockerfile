# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file is used to build CUDA-Q NVQC service container to be deployed to NVCF.
#
# Usage:
# Must be built from the repo root with:
#   DOCKER_BUILDKIT=1 docker build -f docker/release/cudaq.nvqc.Dockerfile . --output out

# Base image is CUDA-Q image 
ARG base_image=nvcr.io/nvidia/nightly/cuda-quantum:latest
FROM $base_image as nvcf_image

ADD tools/cudaq-qpud/nvqc_proxy.py /
ADD tools/cudaq-qpud/json_request_runner.py /

# Launch script: launch cudaq-qpud (nvcf mode) with MPI ranks == Number of NVIDIA GPUs
# IMPORTANT: 
# (1) NVCF function must set container environment variable `NUM_GPUS`
# equal to the number of GPUs on the target platform. This will allow clients to query
# the function capability (number of GPUs) by looking at function info. The below
# entry point script helps prevent mis-configuration by checking that functions are
# created and deployed appropriately.
# (2) NVCF function must set container environment variable `NVQC_REST_PAYLOAD_VERSION` equal
# to the RestRequest payload version with which `cudaq-qpud` in the deployment Docker image was compiled.
# Failure to do so will result in early exits of the entry point command, thus deployment failure.
RUN echo 'cat /opt/nvidia/cudaq/build_info.txt;' \
    'EXPECTED_REST_PAYLOAD_VERSION="$(cudaq-qpud --type nvcf --schema-version | grep -o "CUDA-Q REST API version: \S*" | cut -d ":" -f 2 | tr -d " ")" ;' \
    'if [[ "$NVQC_REST_PAYLOAD_VERSION" !=  "$EXPECTED_REST_PAYLOAD_VERSION" ]]; ' \
    '  then echo "Invalid Deployment: NVQC_REST_PAYLOAD_VERSION environment variable ($NVQC_REST_PAYLOAD_VERSION) does not match cudaq-qpud (expected $EXPECTED_REST_PAYLOAD_VERSION)." && exit 1; fi;' \
    'python3 /nvqc_proxy.py & ' \
    'if [[ "$NUM_GPUS" == "$(nvidia-smi --list-gpus | wc -l)" ]]; then ' \
      'while true; do ' \
        'mpiexec -np $(nvidia-smi --list-gpus | wc -l) cudaq-qpud --type nvcf --port 3031;' \
      'done; ' \
     'else echo "Invalid Deployment: Number of GPUs does not match the hardware" && exit 1; fi' > launch.sh

# Start the cudaq-qpud service
ENTRYPOINT ["bash", "-l", "launch.sh"]
