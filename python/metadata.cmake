# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# We need to be able to access the metadata without loading the cudaq_runtime.
# To do so, we generate a separate (Python) file that is then included in the package.
# This script is used/needed to run it as a command in a custom CMake target, 
# which copies all Python files to the build directory.

if(NOT METADATA_FILE)
    message(FATAL_ERROR "METADATA_FILE is not defined")
endif()

if(CUDA_VERSION_MAJOR)
    file(WRITE ${METADATA_FILE} "cuda_major=${CUDA_VERSION_MAJOR}")
else()
    file(WRITE ${METADATA_FILE} "cuda_major=None")
endif()
