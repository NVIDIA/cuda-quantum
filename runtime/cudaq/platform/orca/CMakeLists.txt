# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-platform-orca)
# find_package(CUDA REQUIRED)
add_library(${LIBRARY_NAME} SHARED OrcaPlatform.cpp 
../common/QuantumExecutionQueue.cpp)
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
      #  $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
       $<INSTALL_INTERFACE:include>
    PRIVATE . ../../)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC 
    cudaq-em-qir 
    cudaq-spin 
    cudaq-common 
  PRIVATE
    pthread
    fmt::fmt-header-only
    # ${CUDA_LIBRARIES}
    cudaq 
    cudaq-platform-default)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-platform-orca-targets DESTINATION lib)

install(EXPORT cudaq-platform-orca-targets
        FILE CUDAQPlatformDefaultTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)

add_target_config(orca)