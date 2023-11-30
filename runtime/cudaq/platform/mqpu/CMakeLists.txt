# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-platform-mqpu)
find_package(CUDA REQUIRED)
add_library(${LIBRARY_NAME} SHARED MultiQPUPlatform.cpp ../common/QuantumExecutionQueue.cpp)
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
       $<INSTALL_INTERFACE:include>
    PRIVATE . ../../)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC 
    cudaq-em-default 
    cudaq-spin 
    cudaq-common 
  PRIVATE 
    pthread
    spdlog::spdlog 
    fmt::fmt-header-only 
    ${CUDA_LIBRARIES})

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-platform-mqpu-targets DESTINATION lib)

add_target_config(nvidia-mqpu)
add_target_config(nvidia-mqpu-fp64)