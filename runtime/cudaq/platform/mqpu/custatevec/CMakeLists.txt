# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
add_library(gpu-emulated-qpu OBJECT GPUEmulatedQPU.cpp)
target_link_libraries(gpu-emulated-qpu PUBLIC 
                        cudaq-common 
                        CUDA::cudart_static
                        spdlog::spdlog 
)
target_include_directories(gpu-emulated-qpu PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>)