# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# ==============================================================================
# Shared library for external consumers (libcudaq-realtime.so)
# ==============================================================================
# This shared library exports a C-compatible host API for wiring dispatchers
# and includes the GPU dispatch kernel device code.

if(CUDA_FOUND)
  set(CUDAQ_REALTIME_SOURCES
    dispatcher/cudaq_realtime_api.cpp
  )

  add_library(cudaq-realtime SHARED ${CUDAQ_REALTIME_SOURCES})
  
  target_include_directories(cudaq-realtime
    PUBLIC
      $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(cudaq-realtime 
    PUBLIC 
      CUDA::cudart_static
    PRIVATE
      cudaq-realtime-host-dispatch
  )

  target_compile_definitions(cudaq-realtime PUBLIC CUDAQ_REALTIME_HAVE_CUDA)

  set_target_properties(cudaq-realtime PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )

  install(TARGETS cudaq-realtime
    COMPONENT realtime-lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  add_library(cudaq-realtime-dispatch STATIC dispatcher/dispatch_kernel.cu)

  target_include_directories(cudaq-realtime-dispatch
    PUBLIC
      $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
  )

  # Link CUDA device runtime library (required for device-side API calls like cudaGraphLaunch)
  find_library(CUDADEVRT_LIBRARY cudadevrt
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    REQUIRED
  )
  
  target_link_libraries(cudaq-realtime-dispatch
    PUBLIC
      CUDA::cudart_static
      ${CUDADEVRT_LIBRARY}
  )

  set_target_properties(cudaq-realtime-dispatch PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )

  install(TARGETS cudaq-realtime-dispatch
    COMPONENT realtime-lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  # ============================================================================
  # Host-side graph dispatcher (optional, for Grace Hopper / Grace Blackwell etc.)
  # ============================================================================
  # Compiled with nvcc so libcu++ (<cuda/std/atomic>) works without extra
  # include paths. Host-only code; no device code in this TU.
  add_library(cudaq-realtime-host-dispatch SHARED
    dispatcher/host_dispatcher.cu
    dispatcher/host_dispatcher_capi.cu
  )

  target_include_directories(cudaq-realtime-host-dispatch
    PUBLIC
      $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(cudaq-realtime-host-dispatch
    PUBLIC
      CUDA::cudart_static
  )

  set_target_properties(cudaq-realtime-host-dispatch PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )

  install(TARGETS cudaq-realtime-host-dispatch
    COMPONENT realtime-lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
