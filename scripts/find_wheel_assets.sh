#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script finds CUDA-Q simulator assets and constructs the CMake
# string argument in the form "<simulator's .so>;<simulator's .yml>;...",
# suitable for wheel build.
#
# Usage:
# bash scripts/find_wheel_assets.sh "$assets"
#
# The assets variable should be set to the path of the directory
# that contains the asset files. 
# Note: it's okay if "$assets" directory does not exist. It will return an empty string.

if [ ! -d $1 ] ; then
    exit 0
fi

for config_file in `find $1/*.yml -maxdepth 0 -type f`; do 
    RESULT_CONFIG="${RESULT_CONFIG:+$RESULT_CONFIG;}${config_file}"; 
done 
for lib_file in `find $1/libnvqir-*.so -maxdepth 0 -type f`; do 
    RESULT_CONFIG="${RESULT_CONFIG:+$RESULT_CONFIG;}${lib_file}"; 
done

echo $RESULT_CONFIG