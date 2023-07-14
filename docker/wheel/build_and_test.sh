#!/bin/bash 

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Build the wheels in a manylinux image, then pass the wheel off to a fresh 
# Ubuntu image where we will install it, then confirm it works by running the
# CUDA-Quantum pytest suite. 
# Usage:
#   bash build_and_test.sh 

#manylinux_image=quay.io/pypa/manylinux_2_28_aarch64
manylinux_image=quay.io/pypa/manylinux_2_28_x86_64
llvm_commit=$(git rev-parse @:tpls/llvm)
docker build -t docker.io/nvidia/cudaq_manylinux_deps:no-zlib -f docker/build/devdeps.manylinux.Dockerfile . \
    --build-arg llvm_commit=$llvm_commit \
    --build-arg manylinux_image=$manylinux_image

rm -rf out/* && \
DOCKER_BUILDKIT=1 \
docker build -t nvidia/cudaq_manylinux_build -f docker/build/cudaq.wheels.Dockerfile . \
    --build-arg release_version=0.4.0 \
    --output out

# docker rmi -f nvidia/cudaq_manylinux_test nvidia/cudaq_manylinux_build 

container_id=`docker run -itd --rm ubuntu:22.04 | grep -e '.*$'`
docker cp out/cuda_quantum-*-manylinux_*_x86_64.whl $container_id:/tmp/
docker cp docs/sphinx/examples/python $container_id:/tmp/
docker attach $container_id
apt-get update && apt-get install -y --no-install-recommends python3 python3-pip
pip install /tmp/cuda_quantum-*-manylinux_*_x86_64.whl --user && python3 -c "import cudaq"
pip install numpy && for file in `ls /tmp/python/`; do python3 /tmp/python/$file; done
exit
