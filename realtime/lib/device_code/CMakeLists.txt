# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# ==============================================================================
# Run-time device link of the unified dispatcher (libcudaq-realtime-device-code.a)
# ==============================================================================
# The nvJitLink linker and the unified dispatch kernel's embedded LTO-IR image,
# linked privately into libcudaq-realtime.so, whose dispatcher API
# (cudaq_dispatcher_get_device_handler) drives them.

if(NOT CUDA_FOUND)
  return()
endif()

# A DEVICE + UNIFIED dispatcher links its kernel, the bridge's hooks and the
# application's handlers at run time.  CUDA-Q's dev images install only the
# runtime library, without the header or the unversioned .so.
if (NOT TARGET CUDA::nvJitLink)
  message(FATAL_ERROR
    "cudaq-realtime needs nvJitLink (header and unversioned library); "
    "install libnvjitlink-dev-${CUDAToolkit_VERSION_MAJOR}-"
    "${CUDAToolkit_VERSION_MINOR}")
endif()

set(CUDAQ_REALTIME_DEVICE_CODE_SOURCES
  nvjitlink_linker.cpp
  unified_dispatch_kernel_image.cpp
)

# The unified dispatch kernel ships as relocatable LTO-IR, embedded in
# libcudaq-realtime.so, and is linked at run time with the bridge's hooks and
# the application's handlers.  An ordinary -rdc source of that library would
# not load: its registration stub needs a device link, which the hooks it
# calls cannot satisfy until run time.  LTO-IR for one architecture links for
# that GPU and every newer one, and the kernel has no __CUDA_ARCH__ branches,
# so a single entry at the oldest supported architecture is enough.
set(CUDAQ_REALTIME_DEVICE_CODE_MIN_ARCH 80 CACHE STRING
  "Oldest GPU architecture the unified dispatch kernel's LTO-IR targets")
set(_kernel_source
  ${CMAKE_CURRENT_SOURCE_DIR}/../daemon/dispatcher/unified_dispatch_kernel.cu)
set(_kernel_fatbin ${CMAKE_CURRENT_BINARY_DIR}/unified_dispatch_kernel.fatbin)
set(_kernel_nvcc_flags)
if (CMAKE_CUDA_HOST_COMPILER)
  list(APPEND _kernel_nvcc_flags -ccbin ${CMAKE_CUDA_HOST_COMPILER})
endif()
add_custom_command(
  OUTPUT ${_kernel_fatbin}
  COMMAND ${CMAKE_CUDA_COMPILER} ${_kernel_nvcc_flags}
    -fatbin -rdc=true -std=c++20
    --generate-code=arch=compute_${CUDAQ_REALTIME_DEVICE_CODE_MIN_ARCH},code=lto_${CUDAQ_REALTIME_DEVICE_CODE_MIN_ARCH}
    -I${CUDAQ_REALTIME_INCLUDE_DIR}
    -MD -MF ${_kernel_fatbin}.d -MT ${_kernel_fatbin}
    ${_kernel_source} -o ${_kernel_fatbin}
  DEPENDS ${_kernel_source}
  DEPFILE ${_kernel_fatbin}.d
  COMMENT "Building the unified dispatch kernel's LTO-IR image"
  VERBATIM)
set_source_files_properties(unified_dispatch_kernel_image.cpp
  PROPERTIES
    COMPILE_DEFINITIONS
      "CUDAQ_REALTIME_UNIFIED_KERNEL_FATBIN=\"${_kernel_fatbin}\""
    OBJECT_DEPENDS ${_kernel_fatbin})
unset(_kernel_source)
unset(_kernel_fatbin)
unset(_kernel_nvcc_flags)

add_library(cudaq-realtime-device-code STATIC
  ${CUDAQ_REALTIME_DEVICE_CODE_SOURCES})

# Same C++ runtime as the rest of libcudaq-realtime.so; see lib/daemon.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set_property(SOURCE ${CUDAQ_REALTIME_DEVICE_CODE_SOURCES} APPEND PROPERTY
    COMPILE_OPTIONS -stdlib=libstdc++)
  if(DEFINED ENV{GCC_TOOLCHAIN} AND IS_DIRECTORY "$ENV{GCC_TOOLCHAIN}")
    set_property(SOURCE ${CUDAQ_REALTIME_DEVICE_CODE_SOURCES} APPEND PROPERTY
      COMPILE_OPTIONS "--gcc-toolchain=$ENV{GCC_TOOLCHAIN}")
  endif()
endif()

target_include_directories(cudaq-realtime-device-code
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
)

target_link_libraries(cudaq-realtime-device-code
  PUBLIC
    CUDA::cudart_static
    CUDA::nvJitLink
)

set_target_properties(cudaq-realtime-device-code PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
