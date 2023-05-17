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
rm -rf _skbuild/
rm -rf cuda_quantum.egg-info/
rm -rf dist/ 
rm -rf MANIFEST.in 

# Build wheel and sdist files out of the python directory,
# as controlled by `python/setup.py`.
# NOTE: Only building the wheel for now, not sdist.
# TODO: The python build package is a requirement for building
# wheels so we should handle its installation. 
# TODO: The build package also requires venv, so we must do a
# `apt-get install python3.10-venv`. This can be avoided, however,
# if we specify `--no-isolation`.
# python3 -m build python/.
python3 -m build --sdist #--wheel

# TODO: auditwheel

# Pip install the wheel.
# python3 -m pip install python/dist/*.whl

# Test if we can import the package.
# python3 -c "import cudaq"