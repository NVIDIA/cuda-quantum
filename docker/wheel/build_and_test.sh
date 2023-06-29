#!/bin/bash 

# Build and test Manylinux Python Wheels run with 
#
# ./build_and_test.sh 

export DOCKER_BUILDKIT=1
# remove old wheels
rm -rf out/*
# build the wheels
docker build -t nvidia/cudaq_manylinux_build --network host . --output out 
# test the wheels 
docker build -t nvidia/cudaq_manylinux_test --network host -f tests/Dockerfile.ubuntu2204 . 
# cleanup
docker rmi -f nvidia/cudaq_manylinux_test nvidia/cudaq_manylinux_build 