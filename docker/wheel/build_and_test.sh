#!/bin/bash 

# Build and test Manylinux Python Wheels run with 
#
# ./build_and_test.sh 

export DOCKER_BUILDKIT=1
# Remove old outputs.
rm -rf out/*
# Build the wheels in a manylinux container.
docker build -t nvidia/cudaq_manylinux_build --network host . --output out
# Test the wheels in a fresh Ubuntu image. This will install the wheel that was built
# in the manylinux container, then run the pytest suite using the cuda-quantum pip package.
docker build -t nvidia/cudaq_manylinux_test --network host -f tests/Dockerfile.ubuntu2204 . 
# Cleanup.
docker rmi -f nvidia/cudaq_manylinux_test nvidia/cudaq_manylinux_build 