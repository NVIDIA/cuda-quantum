# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(NVQLINK_DAEMON_SOURCES
  config.cpp
  daemon.cpp
  dispatcher/cpu_dispatcher.cpp
  dispatcher/dispatcher_factory.cpp
  registry/function_registry.cpp
  registry/rpc_builder.cpp
)

# Add CUDA/GPU sources if CUDA is available
if(CUDA_FOUND)
  list(APPEND NVQLINK_DAEMON_SOURCES
    dispatcher/gpu_dispatcher.cu
  )
endif()

add_library(nvqlink_daemon STATIC ${NVQLINK_DAEMON_SOURCES})

target_include_directories(nvqlink_daemon
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQ_NVQLINK_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(nvqlink_daemon 
  PUBLIC 
    nvqlink_network
    nvqlink_utils
)

if(CUDA_FOUND)
  target_link_libraries(nvqlink_daemon PUBLIC CUDA::cudart)
  target_compile_definitions(nvqlink_daemon PUBLIC NVQLINK_HAVE_CUDA)
endif()

# Add DOCA support if enabled
if(NVQLINK_ENABLE_DOCA)
  target_compile_definitions(nvqlink_daemon PUBLIC NVQLINK_ENABLE_DOCA)
endif()
