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
