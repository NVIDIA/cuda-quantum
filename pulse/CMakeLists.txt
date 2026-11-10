# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# CUDA-Q pulse is a standalone CMake project. It does not build as part of the
# CUDA-Q tree; instead it compiles against the CUDA-Q toolchain shipped in the
# `cudaq-devel` wheel (headers, CMake packages, MLIR/LLVM archives, mlir-tblgen)
# together with the `cudaq` runtime wheel (which provides libcudaqMLIR). Install
# both into the active Python environment and configure this directory directly:
#
#   pip install cudaq cudaq-devel nanobind
#   cmake -S pulse -B build-pulse -G Ninja
#
# See pulse/README.md for the full build, test, and GPU-runtime instructions.
cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

# Set a default build type / install prefix if none was specified. Must be set
# before project().
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel")
set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.cudaq_pulse" CACHE STRING
    "Install path prefix, prepended onto install directories")

# C is enabled because LLVM's HandleLLVMOptions probes the C compiler.
project(cudaq-pulse LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

include(GNUInstallDirs)

# Locate the CUDA-Q CMake packages installed by the `cudaq-devel` wheel. Both
# `cudaq-devel` and `cudaq` unpack into the environment root, so a single prefix
# resolves CUDAQ, MLIR, and LLVM. An explicit -DCMAKE_PREFIX_PATH still wins.
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE _cudaq_pulse_site_packages
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _cudaq_pulse_site_status)
if(_cudaq_pulse_site_status EQUAL 0 AND
   EXISTS "${_cudaq_pulse_site_packages}/lib/cmake/cudaq")
  list(APPEND CMAKE_PREFIX_PATH "${_cudaq_pulse_site_packages}")
  message(STATUS
    "Using CUDA-Q CMake packages from ${_cudaq_pulse_site_packages}/lib/cmake")
endif()

# Pulse never compiles CUDA-Q quantum kernels, so the nvq++ toolchain is not
# required. This must be set before find_package(CUDAQ).
set(CUDAQ_ENABLE_LANGUAGE OFF)
find_package(CUDAQ REQUIRED)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH
     "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(HandleLLVMOptions)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddCUDAQ)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

# add_llvm_executable / add_mlir_library place their artifacts here.
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")

# CUDA-Q installers use a single relocatable library tree. Keep the pulse CMake
# package under that tree on distributions where GNUInstallDirs would otherwise
# default to lib64.
set(CMAKE_INSTALL_LIBDIR lib)

set(CUDAQ_PULSE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CUDAQ_PULSE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(CUDAQ_PULSE_FRONTEND_DIR
    "${CUDAQ_PULSE_SOURCE_DIR}/core/frontend/cudaq_pulse")

include_directories(
  ${CUDAQ_PULSE_SOURCE_DIR}/core/mlir/include
  ${CUDAQ_PULSE_BINARY_DIR}/core/mlir/include)

# Generate the three research-preview dialects.
add_subdirectory(core/mlir/include/cudaq-pulse)

# Build the dialects, transforms, conversions, C API, and optimizer driver.
add_subdirectory(core/mlir)
add_subdirectory(core/mlir/tools/cudaq-pulse-opt)

# Resolve the optional GPU runtime plus the Python and nanobind dependencies.
include(CudaqPulseDependencies)
add_subdirectory(core/mlir/bindings)

# Stage the Python package and register the aggregate build/install target.
include(CudaqPulsePython)

# Keep test and documentation target definitions out of the package entrypoint.
include(CudaqPulseTesting)
include(CudaqPulseDocumentation)
