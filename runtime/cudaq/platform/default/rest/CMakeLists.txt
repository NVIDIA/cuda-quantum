# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "Building REST QPU.")
add_library(cudaq-rest-qpu SHARED RemoteRESTQPU.cpp 
   ../../common/QuantumExecutionQueue.cpp
   Executor.cpp)

add_subdirectory(helpers)

target_include_directories(cudaq-rest-qpu PRIVATE .
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>)

target_link_libraries(cudaq-rest-qpu 
  PUBLIC 
    cudaq-spin 
    cudaq-common 
  PRIVATE 
    cudaq-mlir-runtime 
    MLIRTranslateLib
    fmt::fmt-header-only 
    cudaq 
    cudaq-platform-default)

install(TARGETS cudaq-rest-qpu DESTINATION lib)

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-rest-qpu-targets DESTINATION lib)
