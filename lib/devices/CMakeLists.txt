# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

target_sources(${LIBRARY_NAME} PRIVATE cpu_shmem_device.cpp)

if (CUDA_FOUND) 
  target_sources(${LIBRARY_NAME} PRIVATE cuda_device.cu)
  target_link_libraries(${LIBRARY_NAME}
      PRIVATE 
        CUDA::cudart_static
        CUDA::cuda_driver
    )
endif() 