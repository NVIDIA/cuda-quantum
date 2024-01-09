# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(qpu-emulated-qpu OBJECT GPUEmulatedQPU.cpp)
target_link_libraries(qpu-emulated-qpu PUBLIC 
                        cudaq-common 
                        CUDA::cudart_static
                        spdlog::spdlog 
)