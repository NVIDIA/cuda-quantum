# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# LLVM-dependent library: YAML serialization and runtime argument processing.
add_llvm_library(CUDAQTargetConfigYaml
  StaticConfigYaml.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  LINK_COMPONENTS
  Support
)
target_link_libraries(CUDAQTargetConfigYaml PUBLIC CUDAQTargetConfig)
