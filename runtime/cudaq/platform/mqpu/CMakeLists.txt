# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(helpers)

set(LIBRARY_NAME cudaq-platform-mqpu)
add_library(${LIBRARY_NAME}
  SHARED MultiQPUPlatform.cpp)
target_include_directories(${LIBRARY_NAME}
    PUBLIC
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>
    PRIVATE . ../../)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC
    cudaq-em-default
    cudaq-operator
  PRIVATE
    cudaq-common
    cudaq-logger
    fmt::fmt-header-only
    cudaq
    nvqir
    mqpu_util
    CUDAQTargetConfigUtil
    cudaq-qpu-default
    pthread
)

add_target_config(nvidia-mqpu)

if (cuStateVec_FOUND)
  add_target_config(nvidia-mqpu-mps)
  add_target_config(nvidia-mqpu-fp64)
endif()

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
