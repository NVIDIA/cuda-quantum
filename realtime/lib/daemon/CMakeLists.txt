# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
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
    bridge/bridge_interface_api.cpp
  )

  add_library(cudaq-realtime SHARED ${CUDAQ_REALTIME_SOURCES})
  # The unified dispatch kernel is part of the runtime: the dispatcher calls
  # cudaq_launch_unified_dispatch_device directly, so it must always resolve.
  # Linking the archive here device-links the kernel with its transport hooks
  # (see the target below).
  target_link_libraries(cudaq-realtime PRIVATE cudaq-realtime-unified-dispatch-core)
  # Source-tree CUDA-Q integration uses the same cudaq:: target names
  # that installed exports provide.
  add_library(cudaq::cudaq-realtime ALIAS cudaq-realtime)

  # nvcc compiles the CUDA objects with its GCC host toolchain and libstdc++.
  # Keep the C++ objects and final shared-library link on that same runtime
  # when the surrounding CUDA-Q toolchain uses Clang (the release toolchain
  # otherwise defaults to libc++). The public interface is C-compatible and
  # does not expose C++ standard-library ABI types.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_property(SOURCE ${CUDAQ_REALTIME_SOURCES} APPEND PROPERTY
      COMPILE_OPTIONS -stdlib=libstdc++)
    target_link_options(cudaq-realtime PRIVATE
      "$<HOST_LINK:-stdlib=libstdc++>")
    if(DEFINED ENV{GCC_TOOLCHAIN} AND IS_DIRECTORY "$ENV{GCC_TOOLCHAIN}")
      set_property(SOURCE ${CUDAQ_REALTIME_SOURCES} APPEND PROPERTY
        COMPILE_OPTIONS "--gcc-toolchain=$ENV{GCC_TOOLCHAIN}")
      target_link_options(cudaq-realtime PRIVATE
        "$<HOST_LINK:--gcc-toolchain=$ENV{GCC_TOOLCHAIN}>")
    endif()
  endif()

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
  target_link_options(cudaq-realtime PRIVATE
    "-Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/cudaq-realtime.map")
  set_target_properties(cudaq-realtime PROPERTIES LINK_FLAGS_RELEASE "-s")

  set_target_properties(cudaq-realtime PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )

  install(TARGETS cudaq-realtime
    EXPORT cudaq-realtime-targets
    COMPONENT realtime-lib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  add_library(cudaq-realtime-dispatch STATIC
    dispatcher/dispatch_kernel.cu
  )
  # Source-tree CUDA-Q integration uses the same cudaq:: target names
  # that installed exports provide.
  add_library(cudaq::cudaq-realtime-dispatch ALIAS cudaq-realtime-dispatch)
  set_target_properties(cudaq-realtime-dispatch
    PROPERTIES
    LINK_FLAGS_RELEASE "-Wl,--exclude-libs=ALL")
  set_target_properties(cudaq-realtime-dispatch PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")

  target_include_directories(cudaq-realtime-dispatch
    PUBLIC
      $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
  )

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
    EXPORT cudaq-realtime-targets
    COMPONENT realtime-lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  # ============================================================================
  # Host-side graph dispatcher (optional, for Grace Hopper / Grace Blackwell etc.)
  # ============================================================================
  # Compiled with nvcc so libcu++ (<cuda/std/atomic>) works without extra
  # include paths. Host-only code; no device code in this TU.
  add_library(cudaq-realtime-host-dispatch STATIC
    dispatcher/host_dispatcher.cu
    dispatcher/graph_launch_engine.cu
  )
  # Source-tree CUDA-Q integration uses the same cudaq:: target names
  # that installed exports provide.
  add_library(cudaq::cudaq-realtime-host-dispatch ALIAS cudaq-realtime-host-dispatch)

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
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )

  install(TARGETS cudaq-realtime-host-dispatch
    EXPORT cudaq-realtime-targets
    COMPONENT realtime-lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  # ============================================================================
  # Transport-agnostic unified dispatch kernel
  # ============================================================================
  # The kernel reaches its transport through the __device__ hooks in
  # unified_device_transport.cuh.  CUDA resolves __device__ calls only within
  # a single module, so the hook implementations must be device-linked with
  # the kernel: bridge/gpu_roce appends its own TU to THIS target rather than
  # to its shared library.  That keeps the launcher a runtime artifact --
  # consumers link this archive and hand
  # cudaq_launch_unified_dispatch_device to the dispatcher, exactly as they
  # link cudaq-realtime-dispatch and hand over
  # cudaq_launch_dispatch_kernel_regular.
  #
  # Splitting the transport into its own TU is what keeps the kernel free of
  # DOCA today; sources here must not include transport headers.  Once the
  # data plane is genuinely pluggable the transport TU leaves this target
  # again, and the include isolation can go back to being build-enforced.
  add_library(cudaq-realtime-unified-dispatch-core STATIC
    dispatcher/unified_dispatch_kernel.cu
  )
  add_library(cudaq::cudaq-realtime-unified-dispatch-core
    ALIAS cudaq-realtime-unified-dispatch-core)

  target_include_directories(cudaq-realtime-unified-dispatch-core
    PUBLIC
      $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(cudaq-realtime-unified-dispatch-core
    PUBLIC
      CUDA::cudart_static
      ${CUDADEVRT_LIBRARY}
  )

  set_target_properties(cudaq-realtime-unified-dispatch-core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )

  install(TARGETS cudaq-realtime-unified-dispatch-core
    EXPORT cudaq-realtime-targets
    COMPONENT realtime-lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  if (CUDAQ_REALTIME_ENABLE_HSB_TOOLS)
    add_subdirectory(bridge/gpu_roce)
  endif()
  # CPU transport bridge providers (thin provider .so's over the
  # cpu_transport ring transceivers; loaded via CUDAQ_REALTIME_BRIDGE_LIB).
  add_subdirectory(bridge/udp)
  add_subdirectory(bridge/cpu_roce)
endif()
