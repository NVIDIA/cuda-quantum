# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include_guard(DIRECTORY)

# Build-time dependency discovery for pulse: the optional GPU runtime, Python,
# and nanobind. MLIR/LLVM and the CUDA-Q CMake packages are resolved in the
# top-level CMakeLists.txt from the installed cudaq/cudaq-devel wheels.

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
  else()
    # Only probe opportunistically when a CUDA toolkit is actually present;
    # FindcuDensityMat requires CUDAToolkit to resolve its library suffixes.
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      find_package(cuDensityMat QUIET)
    endif()
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

# nanobind resolves its Python dependency through FindPython, whereas MLIR and
# the top-level project use FindPython3. Seed the former from the latter so both
# modules agree on a single interpreter.
if(NOT Python_EXECUTABLE)
  set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
endif()
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# nanobind ships its own CMake package inside the Python wheel. Ask the
# interpreter where it is, while still letting -Dnanobind_DIR override.
if(NOT nanobind_DIR)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE _cudaq_pulse_nanobind_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _cudaq_pulse_nanobind_status
    ERROR_QUIET)
  if(NOT _cudaq_pulse_nanobind_status EQUAL 0)
    message(FATAL_ERROR
      "nanobind was not found in the active Python environment. Install it "
      "with `pip install \"nanobind>=2.12\"`, or point at an existing CMake "
      "package with -Dnanobind_DIR=<dir>.")
  endif()
  set(nanobind_DIR "${_cudaq_pulse_nanobind_dir}")
endif()
find_package(nanobind CONFIG REQUIRED)
