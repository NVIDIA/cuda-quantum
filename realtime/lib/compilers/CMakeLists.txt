# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CUDAQ_LIBRARY_MODE ON)
find_package(CUDAQ QUIET)
if (CUDAQ_FOUND)
  message(STATUS "CUDA-Q found: Enabling CUDA-Q compiler support")
  add_subdirectory(cudaq)
else()
  message(STATUS "CUDA-Q not found: Disabling CUDA-Q compiler support (optional)")
endif()