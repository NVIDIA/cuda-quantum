# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# LLVM-free library: config data structures and helpers.
add_library(CUDAQTargetConfig
  TargetConfig.cpp
  TargetPluginLibrary.cpp
)

target_include_directories(CUDAQTargetConfig PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cudaq/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(CUDAQTargetConfig PRIVATE ${CMAKE_DL_LIBS})

install(TARGETS CUDAQTargetConfig
        EXPORT cudaq-common-targets
        DESTINATION lib
        COMPONENT Runtime)

add_library(CUDAQTargetDatabase STATIC TargetDatabase.cpp)
target_include_directories(CUDAQTargetDatabase
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cudaq/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(CUDAQTargetDatabase PUBLIC CUDAQTargetConfig)
install(TARGETS CUDAQTargetDatabase
        EXPORT cudaq-common-targets
        DESTINATION lib
        COMPONENT Runtime)

add_subdirectory(Yaml)
