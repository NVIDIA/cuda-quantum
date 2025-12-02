# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(NVQLINK_NETWORK_SOURCES
  memory/buffer.cpp
  memory/memory_pool.cpp
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

add_library(nvqlink_network STATIC ${NVQLINK_NETWORK_SOURCES})

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