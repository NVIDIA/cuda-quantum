# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.   #
# ============================================================================ #

# Simple external bridge (implementing the bridge interface )

add_library(simple_bridge SHARED simple_bridge_impl.cpp)
target_include_directories(simple_bridge
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(simple_bridge PRIVATE CUDA::cudart)

# Bridge executable (.cpp, linked with CUDA)
add_executable(simple_app simple_app.cpp init_rpc_increment_function_table.cu)

set_target_properties(simple_app PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(simple_app
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})

# Link order: static archives first, then shared
# Note: no direct dependency on hololink needed.
target_link_libraries(simple_app
  PRIVATE
    cudaq-realtime-dispatch
    cudaq-realtime)


# configure_file(hololink_test.sh.in hololink_test.sh @ONLY)
