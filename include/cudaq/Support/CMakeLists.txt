# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if (DEFINED ENV{CUDA_QUANTUM_VERSION})
  # The version was defined by the user (likely a bot performing the build), so
  # use the value provided as is.
  set(CUDA_QUANTUM_VERSION "$ENV{CUDA_QUANTUM_VERSION}")
else()
  # Otherwise, create a version based on the nearest tag in the git repo.
  execute_process(COMMAND git describe --tags --abbrev=0 --dirty=-developer
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE CUDA_QUANTUM_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

configure_file("Version.h.in" "Version.h" @ONLY)
