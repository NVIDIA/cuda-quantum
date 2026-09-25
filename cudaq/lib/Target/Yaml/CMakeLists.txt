# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# LLVM-dependent library: YAML serialization and runtime argument processing.
add_library(CUDAQTargetConfigYaml
  TargetConfigYaml.cpp
  TargetCatalog.cpp
)
target_include_directories(CUDAQTargetConfigYaml
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cudaq/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
# Link MLIR dynamically (default) or statically (if CUDAQ_MLIR_STATIC is set).
target_link_libraries(CUDAQTargetConfigYaml PUBLIC CUDAQTargetConfig
  "$<IF:$<BOOL:$<TARGET_PROPERTY:CUDAQ_MLIR_STATIC>>,cudaq-mlir-static,cudaqMLIR>")
target_compile_options(CUDAQTargetConfigYaml PRIVATE "-fexceptions" "-frtti")

install(TARGETS CUDAQTargetConfigYaml
        EXPORT cudaq-common-targets
        DESTINATION lib
        COMPONENT Runtime)
