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
# application's handlers at run time with nvJitLink.  libcudaq-realtime.so opens
# libnvJitLink.so.<major> only when such a dispatcher first links, so a host
# that never does (pure host dispatch, no GPU) loads without it.  Building
# needs just the nvJitLink header, which a default CUDA toolkit install may
# lack; the dev images install libnvjitlink-dev.
find_path(CUDAQ_REALTIME_NVJITLINK_INCLUDE_DIR nvJitLink.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS})
if (NOT CUDAQ_REALTIME_NVJITLINK_INCLUDE_DIR)
  message(FATAL_ERROR
    "cudaq-realtime needs the nvJitLink header; "
    "install libnvjitlink-dev-${CUDAToolkit_VERSION_MAJOR}-"
    "${CUDAToolkit_VERSION_MINOR}")
endif()

# The header's inline wrappers call the versioned entry points of the release
# it ships with, __nvJitLink<Name>_<major>_<minor>; the run-time lookup asks
# the library for that same version.
file(STRINGS "${CUDAQ_REALTIME_NVJITLINK_INCLUDE_DIR}/nvJitLink.h"
  _nvjitlink_create REGEX "__nvJitLinkCreate_[0-9]+_[0-9]+")
string(REGEX MATCH "__nvJitLinkCreate_([0-9]+)_([0-9]+)" _nvjitlink_create
  "${_nvjitlink_create}")
if (NOT _nvjitlink_create)
  message(FATAL_ERROR
    "cannot find the nvJitLink ABI version in "
    "${CUDAQ_REALTIME_NVJITLINK_INCLUDE_DIR}/nvJitLink.h")
endif()
set(_nvjitlink_major ${CMAKE_MATCH_1})
set(_nvjitlink_minor ${CMAKE_MATCH_2})
set_property(SOURCE nvjitlink_linker.cpp APPEND PROPERTY COMPILE_DEFINITIONS
  "CUDAQ_REALTIME_NVJITLINK_ABI=${_nvjitlink_major}_${_nvjitlink_minor}"
  "CUDAQ_REALTIME_NVJITLINK_VERSION=\"${_nvjitlink_major}.${_nvjitlink_minor}\""
  "CUDAQ_REALTIME_NVJITLINK_SONAME=\"libnvJitLink.so.${_nvjitlink_major}\"")
unset(_nvjitlink_create)
unset(_nvjitlink_major)
unset(_nvjitlink_minor)

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
  PRIVATE
    ${CUDAQ_REALTIME_NVJITLINK_INCLUDE_DIR}
)

target_link_libraries(cudaq-realtime-device-code
  PUBLIC
    CUDA::cudart_static
  PRIVATE
    ${CMAKE_DL_LIBS}
)

set_target_properties(cudaq-realtime-device-code PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
