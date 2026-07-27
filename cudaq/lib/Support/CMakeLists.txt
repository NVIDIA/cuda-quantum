# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_library(CUDAQSupport
  Plugin.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Pass

  DEPENDS
  mlir-generic-headers)

# Link MLIR dynamically (default) or statically (if CUDAQ_MLIR_STATIC is set).
target_link_libraries(CUDAQSupport PUBLIC
  "$<IF:$<BOOL:$<TARGET_PROPERTY:CUDAQ_MLIR_STATIC>>,cudaq-mlir-static,cudaqMLIR>")
