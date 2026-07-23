# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Minimal out-of-tree example that builds a CUDA-Q MLIR Python extension against
# an installed cudaq-devel wheel. The script `scripts/validate_devel_wheel.sh`
# validates the devel wheel against this example.
#
# Layout follows MLIR conventions:
#
#   include/Trivial/  - public headers for the dialect + pass
#   lib/              - the pure MLIR library
#   python/           - the nanobind extension built with the CUDA-Q helpers

cmake_minimum_required(VERSION 3.30)
project(cudaq_mlir_extension LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the CUDA-Q installation directory from the Python site-packages.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE _cudaq_site_packages
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _cudaq_site_status)
if(_cudaq_site_status EQUAL 0 AND EXISTS "${_cudaq_site_packages}/lib/cmake/cudaq")
  list(APPEND CMAKE_PREFIX_PATH "${_cudaq_site_packages}")
  message(STATUS "Using CUDA-Q CMake packages from ${_cudaq_site_packages}/lib/cmake")
endif()

find_package(CUDAQ REQUIRED)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")

include(HandleLLVMOptions)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(MLIRDetectPythonEnv)
include(AddMLIRPython)
include(AddCUDAQ)

include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include)

# include/ runs mlir-tblgen (from the devel wheel) to generate the dialect, op,
# and pass headers; lib/ and python/ consume them.
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)
