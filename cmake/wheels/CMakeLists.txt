# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Native libraries, headers, and exports for the core wheel.
# Bundle the shared OpenMP runtime under its normal SONAME so other wheels can
# exclude it from repair and use core's copy without dependency-name rewriting.
foreach(_omp_name IN LISTS OpenMP_CXX_LIB_NAMES)
  get_filename_component(_omp_library "${OpenMP_${_omp_name}_LIBRARY}" REALPATH)
  if(_omp_library MATCHES "\\.so(\\.[0-9]+)*$|\\.dylib$")
    get_filename_component(_omp_directory "${_omp_library}" DIRECTORY)
    get_filename_component(_omp_basename "${_omp_library}" NAME_WE)
    file(GLOB _omp_files "${_omp_directory}/${_omp_basename}.so*"
      "${_omp_directory}/${_omp_basename}*.dylib")
    install(FILES ${_omp_files} DESTINATION lib COMPONENT Core)
  endif()
endforeach()

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
