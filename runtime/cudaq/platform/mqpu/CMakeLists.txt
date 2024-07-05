# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_subdirectory(helpers)

if (CUDA_FOUND AND CUSTATEVEC_ROOT)
  add_subdirectory(custatevec)
endif()
add_subdirectory(remote)

set(LIBRARY_NAME cudaq-platform-mqpu)
add_library(${LIBRARY_NAME}
  SHARED MultiQPUPlatform.cpp ../common/QuantumExecutionQueue.cpp)
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<INSTALL_INTERFACE:include>
    PRIVATE . ../../)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC 
    cudaq-em-default 
    cudaq-spin 
    cudaq-common 
  PRIVATE 
    cudaq
    mqpu_util
    pthread
    spdlog::spdlog 
    fmt::fmt-header-only 
)

if (CUDA_FOUND AND CUSTATEVEC_ROOT)
  target_link_libraries(${LIBRARY_NAME} PRIVATE gpu-emulated-qpu)
  add_target_config(nvidia-mqpu)
  add_target_config(nvidia-mqpu-mps)
  add_target_config(nvidia-mqpu-fp64)
endif()
  
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
install(TARGETS ${LIBRARY_NAME}
  EXPORT cudaq-platform-mqpu-targets DESTINATION lib)
add_target_config(remote-mqpu)
add_target_config(nvqc)
