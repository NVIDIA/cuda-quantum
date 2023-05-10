#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# NOTE: This file is expected to be executed from the outermost
# directory, `cuda-quantum/`.

# NOTE: As-is, the paths are hard-coded to the docker image I've been
# using on my machine. These may need to be modified if you're looking
# to run this script -- until I can abstract the paths in setup.py away.

# Remove previous build outputs. 
rm -rf python/_skbuild/
rm -rf python/cuda_quantum.egg-info/
rm -rf python/dist/ 
rm -rf python/MANIFEST.in 

# Build wheel and sdist files out of the python directory,
# as controlled by `python/setup.py`.
python3 -m build python/.