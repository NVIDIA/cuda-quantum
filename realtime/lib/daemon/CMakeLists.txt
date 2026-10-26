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
# This shared library exports a C-compatible host API for wiring dispatchers.
# The GPU dispatch kernels live in libcudaq-realtime-dispatch.a, which
# consumers link themselves.

if(CUDA_FOUND)
  set(CUDAQ_REALTIME_SOURCES
    dispatcher/cudaq_realtime_api.cpp
    bridge/bridge_interface_api.cpp
  )

  add_library(cudaq-realtime SHARED ${CUDAQ_REALTIME_SOURCES})
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
    dispatcher/unified_dispatch_kernel.cu
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

  if (CUDAQ_REALTIME_ENABLE_HSB_TOOLS)
    add_subdirectory(bridge/gpu_roce)
  endif()

  # No transport supplied the unified device hooks, so link the null data
  # plane; see unified_transport_none.cu.
  get_property(_cudaq_unified_transport GLOBAL
    PROPERTY CUDAQ_REALTIME_UNIFIED_TRANSPORT_TU)
  if (NOT _cudaq_unified_transport)
    set(_cudaq_unified_transport dispatcher/unified_transport_none.cu)
    target_sources(cudaq-realtime-dispatch PRIVATE ${_cudaq_unified_transport})
  endif()

  # Device LTO for the unified kernel and its transport hooks.  The hooks are
  # extern __device__ functions in another TU, so without LTO they stay
  # out-of-line calls on the per-message path.  Both TUs carry LTO IR next to
  # their SASS -- nvcc rejects a bare -dlto beside the global -gencode flags
  # -- and consumers device-link with -dlto, which inlines the hooks into the
  # kernel.  A consumer linking without -dlto still gets the plain SASS.
  # nvlink sizes the kernel's registers for every handler reachable through
  # the function table, LTO or not, since they share the consumer's module.
  foreach(_arch IN LISTS CUDA_TARGET_ARCHS)
    set_property(SOURCE dispatcher/unified_dispatch_kernel.cu
      ${_cudaq_unified_transport} APPEND PROPERTY COMPILE_OPTIONS
      "--generate-code=arch=compute_${_arch},code=lto_${_arch}")
  endforeach()
  unset(_arch)
  if (CUDA_TARGET_ARCHS)
    target_link_options(cudaq-realtime-dispatch INTERFACE
      "$<DEVICE_LINK:-dlto>")
  endif()
  unset(_cudaq_unified_transport)
  # CPU transport bridge providers (thin provider .so's over the
  # cpu_transport ring transceivers; loaded via CUDAQ_REALTIME_BRIDGE_LIB).
  add_subdirectory(bridge/udp)
  add_subdirectory(bridge/cpu_roce)
endif()
