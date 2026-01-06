# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(mqpu_util OBJECT MQPUUtils.cpp)
target_link_libraries(mqpu_util
    PRIVATE
        LLVMSupport
        cudaq-common  
        spdlog::spdlog 
        fmt::fmt-header-only
)

if (CUDA_FOUND AND CUSTATEVEC_ROOT)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(mqpu_util PRIVATE CUDAQ_ENABLE_CUDA)
  target_link_libraries(mqpu_util PRIVATE CUDA::cudart_static)
  target_include_directories(mqpu_util PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)
endif()