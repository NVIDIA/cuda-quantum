# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-fermioniq-qpu)
message(STATUS "Building Fermioniq REST QPU.")

add_library(${LIBRARY_NAME} SHARED FermioniqQPU.cpp)

target_include_directories(${LIBRARY_NAME} PRIVATE .
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC 
    cudaq-operator
    cudaq-common 
  PRIVATE
    pthread
    cudaq-mlir-runtime 
    fmt::fmt-header-only
    cudaq 
    cudaq-platform-default)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)

# install(EXPORT cudaq-fermioniq-qpu-targets
#         FILE CUDAQQPUFermioniqTargets.cmake
#         NAMESPACE cudaq::fermioniq::
#         DESTINATION lib/cmake/cudaq)

add_target_config(fermioniq)

add_subdirectory(helpers)
