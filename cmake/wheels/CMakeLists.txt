# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Search hints for resolving OpenMP's runtime dependencies during wheel staging.
# OpenMP's link inputs can include linker scripts and system libraries such as
# pthread; only the actual omp/gomp runtime belongs in the wheel.
set(CUDAQ_WHEEL_OPENMP_DIRS)
if(NOT APPLE)
  foreach(_omp_name IN LISTS OpenMP_CXX_LIB_NAMES)
    if(_omp_name MATCHES "^(gomp|omp)$")
      get_filename_component(_omp_directory "${OpenMP_${_omp_name}_LIBRARY}" DIRECTORY)
      list(APPEND CUDAQ_WHEEL_OPENMP_DIRS "${_omp_directory}")
      get_filename_component(_omp_library "${OpenMP_${_omp_name}_LIBRARY}" REALPATH)
      get_filename_component(_omp_directory "${_omp_library}" DIRECTORY)
      list(APPEND CUDAQ_WHEEL_OPENMP_DIRS "${_omp_directory}")
    endif()
  endforeach()
  list(APPEND CUDAQ_WHEEL_OPENMP_DIRS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})
  list(REMOVE_DUPLICATES CUDAQ_WHEEL_OPENMP_DIRS)
endif()

# Native libraries, headers, and exports for the core wheel.
install(EXPORT cudaq-core-targets FILE CUDAQCoreTargets.cmake
  NAMESPACE cudaq:: DESTINATION lib/cmake/cudaq COMPONENT Core)
install(FILES
  ../modules/CUDAQCommonConfig.cmake
  ../modules/CUDAQOperatorConfig.cmake
  ../modules/CUDAQLoggerConfig.cmake
  ../modules/CUDAQPythonBindingsConfig.cmake
  DESTINATION lib/cmake/cudaq COMPONENT Core)
# Component installs select the core-only package configuration; full native
# installs use cmake/modules/CUDAQConfig.cmake with frontend targets.
install(FILES CUDAQConfig.cmake DESTINATION lib/cmake/cudaq
  COMPONENT Core EXCLUDE_FROM_ALL)
install(FILES
  "${CMAKE_BINARY_DIR}/cmake/modules/NVQIRConfig.cmake"
  ../modules/NVQIRTargetConfigGen.cmake
  DESTINATION lib/cmake/nvqir COMPONENT Core)
# CompileTarget and runtime backend headers use these LLVM-free declarations.
install(FILES
  "${CMAKE_SOURCE_DIR}/cudaq/include/cudaq/Target/TargetConfig.h"
  DESTINATION include/cudaq/Target COMPONENT Core)
install(FILES
  "${CMAKE_SOURCE_DIR}/cudaq/include/cudaq/Support/Hash.h"
  "${CMAKE_SOURCE_DIR}/cudaq/include/cudaq/Support/Tuple.h"
  DESTINATION include/cudaq/Support COMPONENT Core)
set(CUDAQ_WHEEL_BUILD_DIR "${CMAKE_BINARY_DIR}")
set(CUDAQ_WHEEL_KIND core)
configure_file(InstallWheel.cmake.in InstallCoreWheel.cmake @ONLY)
# Wheel-specific paths are staged only when WheelCore is explicitly selected.
install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/InstallCoreWheel.cmake"
  COMPONENT WheelCore EXCLUDE_FROM_ALL)
