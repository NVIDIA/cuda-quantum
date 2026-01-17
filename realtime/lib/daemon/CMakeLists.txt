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

# ==============================================================================
# Shared library for external consumers (libcudaq-realtime.so)
# ==============================================================================
# This shared library exports a C-compatible host API for wiring dispatchers.
# The GPU dispatch kernel itself is header-only (dispatch_kernel.cuh) and must
# be compiled into the consumer's TU alongside the decoder device code.

if(CUDA_FOUND)
  set(CUDAQ_REALTIME_SOURCES
    dispatcher/cudaq_realtime_api.cpp
  )

  add_library(cudaq-realtime SHARED ${CUDAQ_REALTIME_SOURCES})
  
  target_include_directories(cudaq-realtime
    PUBLIC
      $<BUILD_INTERFACE:${CUDAQ_NVQLINK_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(cudaq-realtime 
    PUBLIC 
      CUDA::cudart
  )

  target_compile_definitions(cudaq-realtime PUBLIC NVQLINK_HAVE_CUDA)

  set_target_properties(cudaq-realtime PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )

  install(TARGETS cudaq-realtime
    COMPONENT realtime-lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

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
