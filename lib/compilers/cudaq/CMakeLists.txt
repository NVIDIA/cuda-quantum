# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cudaq_config.h.in"
               "${CMAKE_BINARY_DIR}/include/compilers/cudaq/cudaq_config.h"
               @ONLY)
target_include_directories(cudaq-qclink PRIVATE ${CMAKE_BINARY_DIR}/include/compilers/cudaq)
target_sources(cudaq-qclink PRIVATE cudaq.cpp cudaq_toolchain.cpp)