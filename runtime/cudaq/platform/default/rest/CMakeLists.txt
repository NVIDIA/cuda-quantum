# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

message(STATUS "Building REST QPU.")
add_library(cudaq-rest-qpu SHARED RemoteRESTQPU.cpp)

add_subdirectory(helpers)

target_include_directories(cudaq-rest-qpu PRIVATE .
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>)

# The AWS SDK belongs to `cudaq-serverhelper-braket` alone. See the note in
# helpers/braket/CMakeLists.txt on why it must not be linked twice.

target_link_libraries(cudaq-rest-qpu
  PUBLIC
    cudaq-operator
    cudaq-common
  PRIVATE
    cudaq-mlir-runtime
    fmt::fmt-header-only
    cudaq
    cudaq-platform-default
    nvqir
    ZLIB::ZLIB
    cudaqMLIR)

install(TARGETS cudaq-rest-qpu DESTINATION lib COMPONENT Runtime)
