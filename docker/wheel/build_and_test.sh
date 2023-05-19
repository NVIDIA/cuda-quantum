#!/bin/bash 

# Build and test Manylinux Python Wheels
# run with (use your private git ssh ke)
#
# ./build_and_test.sh ~/.ssh/id_ed25519

export DOCKER_BUILDKIT=1
# remove old wheels
rm -rf out/*
# build the wheels
docker build -t nvidia/cudaq_manylinux_build --network host --build-arg SSH_PRIVATE_KEY="$(cat $1)" . --output out 
# test the wheels 
docker build -t nvidia/cudaq_manylinux_test --network host -f tests/Dockerfile.ubuntu2204 . 
# cleanup
docker rmi -f nvidia/cudaq_manylinux_test nvidia/cudaq_manylinux_build 