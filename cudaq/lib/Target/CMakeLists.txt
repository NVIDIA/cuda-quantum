# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# LLVM-free library: config data structures and helpers.
add_library(CUDAQTargetConfig
  StaticConfig.cpp
)
target_include_directories(CUDAQTargetConfig PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cudaq/include>
  $<INSTALL_INTERFACE:include>
)
install(TARGETS CUDAQTargetConfig EXPORT cudaq-targets DESTINATION lib)

add_subdirectory(Yaml)
