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

# Remove previous build outputs. 
rm -rf python/_skbuild/
rm -rf python/cuda_quantum.egg-info/
rm -rf python/dist/ 
rm -rf ../python/MANIFEST.in 

# Build wheel and sdist files out of the python directory,
# as controlled by `python/setup.py`.
python3 -m build python/.