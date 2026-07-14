# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-platform-qdmi)

find_package(mqt-core CONFIG REQUIRED)
if (NOT TARGET MQT::CoreFoMaC)
  message(FATAL_ERROR
    "The QDMI backend requires the MQT::CoreFoMaC target from MQT Core.")
endif()

message(STATUS "Building QDMI platform.")

add_library(${LIBRARY_NAME}
  SHARED
    QDMIPlatform.cpp
    QDMIQPU.cpp
)

target_include_directories(${LIBRARY_NAME} PRIVATE .
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC
    cudaq-operator
    cudaq-common
  PRIVATE
    cudaq-logger
    cudaq-mlir-runtime
    fmt::fmt-header-only
    cudaq
    CUDAQTargetConfigYaml
    MQT::CoreFoMaC
    nvqir
)

add_target_config(qdmi)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
