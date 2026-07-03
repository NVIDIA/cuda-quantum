# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.28)
project(cudaq-pulse LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Find MLIR (from installed cudaq-mlir wheel or standalone LLVM build)
# ---------------------------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIR at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/core/mlir/include)
include_directories(${PROJECT_BINARY_DIR}/core/mlir/include)

# ---------------------------------------------------------------------------
# Tablegen targets for our three dialects
# ---------------------------------------------------------------------------
add_subdirectory(core/mlir/include/cudaq-pulse)

# ---------------------------------------------------------------------------
# MLIR dialects, transforms, conversions, CAPI
# ---------------------------------------------------------------------------
add_subdirectory(core/mlir)

# ---------------------------------------------------------------------------
# Tools (cudaq-pulse-opt for FileCheck testing)
# ---------------------------------------------------------------------------
add_subdirectory(core/mlir/tools/cudaq-pulse-opt)

# ---------------------------------------------------------------------------
# C++ runtime (cuDensityMat)
# ---------------------------------------------------------------------------
add_subdirectory(core/runtime)

# ---------------------------------------------------------------------------
# Lit / FileCheck testing
# ---------------------------------------------------------------------------
set(CUDAQ_PULSE_TOOLS_DIR "${CMAKE_BINARY_DIR}/core/mlir/tools/cudaq-pulse-opt")
set(CUDAQ_PULSE_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
configure_file(
  ${CMAKE_SOURCE_DIR}/test/lit.site.cfg.py.in
  ${CMAKE_BINARY_DIR}/test/lit.site.cfg.py
  @ONLY
)

# ---------------------------------------------------------------------------
# Python bindings (nanobind extension) — optional
# ---------------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG QUIET)
if(Python_FOUND AND nanobind_FOUND)
  add_subdirectory(core/mlir/bindings)
else()
  message(STATUS "Skipping Python bindings (nanobind not found)")
endif()
