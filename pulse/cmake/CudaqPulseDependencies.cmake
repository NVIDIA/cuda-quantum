# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard(DIRECTORY)

# Build-time dependency discovery for the pulse source-tree build (GPU runtime,
# Python, nanobind). This is distinct from the installed package config: a
# downstream find_package(cudaq-pulse CONFIG) consumer re-discovers the runtime
# dependencies through cmake/cudaq-pulse-config.cmake.in instead.

# Enable the optional GPU runtime whenever cuDensityMat is available. An
# explicitly provided SDK root is a request, so fail instead of silently
# falling back to the compiler-only package when that root is invalid.
set(_cudaq_pulse_cudm_requested FALSE)
if(CUDENSITYMAT_ROOT OR cuDensityMat_ROOT OR
   NOT "$ENV{CUDENSITYMAT_ROOT}" STREQUAL "" OR
   NOT "$ENV{cuDensityMat_ROOT}" STREQUAL "")
  set(_cudaq_pulse_cudm_requested TRUE)
endif()

if(NOT TARGET cuDensityMat::cuDensityMat)
  if(_cudaq_pulse_cudm_requested)
    find_package(cuDensityMat REQUIRED)
  elseif(CUDA_FOUND)
    find_package(cuDensityMat QUIET)
  endif()
endif()

if(TARGET cuDensityMat::cuDensityMat)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS
    "CUDA-Q pulse GPU runtime enabled with cuDensityMat ${cuDensityMat_VERSION}")
  add_subdirectory(core/runtime)
else()
  message(STATUS
    "CUDA-Q pulse GPU runtime disabled (cuDensityMat was not found)")
endif()

# Match CUDA-Q's source-build behavior: use its vendored nanobind checkout by
# default, while allowing nanobind_DIR to select an installed CMake package
# such as the one shipped in the nanobind Python wheel.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
if(NOT nanobind_DIR)
  set(nanobind_DIR "${CMAKE_SOURCE_DIR}/tpls/nanobind/cmake")
endif()
find_package(nanobind CONFIG REQUIRED)
