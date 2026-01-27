# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(NVQLINK_NETWORK_SOURCES
  memory/buffer.cpp
  serialization/input_stream.cpp
  serialization/output_stream.cpp
)

# Add CUDA/GPU sources if CUDA is available
if(CUDA_FOUND)
  list(APPEND NVQLINK_NETWORK_SOURCES
    channels/gpu_channel.cu
    serialization/gpu_input_stream.cu
    serialization/gpu_output_stream.cu
  )
endif()

# Add RoCE sources if enabled
if(NVQLINK_ENABLE_ROCE)
  find_library(IBVERBS_LIB ibverbs REQUIRED)
  
  message(STATUS "RoCE channel enabled")
  message(STATUS "  libibverbs library: ${IBVERBS_LIB}")

  list(APPEND NVQLINK_NETWORK_SOURCES 
    channels/roce/roce_channel.cpp
    channels/roce/roce_buffer_pool.cpp
    channels/roce/verbs_context.cpp
    steering/verbs_flow_switch.cpp
  )
endif()

# Add DOCA sources if enabled
if(NVQLINK_ENABLE_DOCA)
  # Require CUDA for DOCA
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "DOCA channel requires CUDA support. Enable CUDA or disable DOCA.")
  endif()
  
  # Find DOCA libraries using pkg-config
  find_package(PkgConfig REQUIRED)
  
  pkg_check_modules(DOCA_COMMON REQUIRED IMPORTED_TARGET doca-common)
  pkg_check_modules(DOCA_GPUNETIO REQUIRED IMPORTED_TARGET doca-gpunetio)
  pkg_check_modules(DOCA_VERBS REQUIRED IMPORTED_TARGET doca-verbs)
  pkg_check_modules(DOCA_RDMA REQUIRED IMPORTED_TARGET doca-rdma)
  
  # Also need libibverbs and libmlx5
  if(NOT IBVERBS_LIB)
    find_library(IBVERBS_LIB ibverbs REQUIRED)
  endif()
  find_library(MLX5_LIB mlx5 REQUIRED)
  
  message(STATUS "DOCA GPUNetIO channel enabled")
  message(STATUS "  doca-common: ${DOCA_COMMON_LIBRARIES}")
  message(STATUS "  doca-gpunetio: ${DOCA_GPUNETIO_LIBRARIES} (includes doca_gpu types)")
  message(STATUS "  doca-verbs: ${DOCA_VERBS_LIBRARIES}")
  message(STATUS "  doca-rdma: ${DOCA_RDMA_LIBRARIES}")
  message(STATUS "  ibverbs: ${IBVERBS_LIB}")
  message(STATUS "  mlx5: ${MLX5_LIB}")
  
  list(APPEND NVQLINK_NETWORK_SOURCES
    channels/doca/doca_channel.cpp
    channels/doca/doca_completion_queue.cpp
    channels/doca/doca_queue_pair.cpp
    channels/doca/doca_buffer_pool.cpp
    channels/doca/doca_rpc_kernel.cu
  )
endif()

add_library(nvqlink_network STATIC ${NVQLINK_NETWORK_SOURCES})

# Enable CUDA separable compilation for device function pointers
# to work across compilation units (e.g., gpu_functions.cu <-> doca_rpc_kernel.cu)
if(CUDA_FOUND)
  set_target_properties(nvqlink_network PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

target_include_directories(nvqlink_network
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQ_NVQLINK_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(nvqlink_network PUBLIC nvqlink_utils)

if(CUDA_FOUND)
  target_link_libraries(nvqlink_network PUBLIC CUDA::cudart)
endif()

if(NVQLINK_ENABLE_ROCE)
  target_link_libraries(nvqlink_network PUBLIC ${IBVERBS_LIB})
endif()

if(NVQLINK_ENABLE_DOCA)
  target_link_libraries(nvqlink_network PUBLIC
    PkgConfig::DOCA_COMMON
    PkgConfig::DOCA_GPUNETIO
    PkgConfig::DOCA_VERBS
    PkgConfig::DOCA_RDMA
    ${IBVERBS_LIB}
    ${MLX5_LIB}
  )
endif()