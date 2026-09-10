# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.30)

# Refuse to configure in-source.  An in-source configure scatters
# CMakeFiles/, Makefile, cmake_install.cmake, etc. through every
# subdirectory that has a CMakeLists.txt -- see build/ for the expected
# out-of-source layout.  Use `cmake -S . -B build` (or any other build
# directory that isn't the source tree).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not allowed.\n"
    "  Source dir: ${CMAKE_SOURCE_DIR}\n"
    "  Build  dir: ${CMAKE_BINARY_DIR}\n"
    "Run 'cmake -S . -B build [options]' from the repo root instead.\n"
    "If you have already run cmake here, remove the stray files first:\n"
    "  rm -rf CMakeCache.txt CMakeFiles/ _deps/ Makefile cmake_install.cmake\n"
    "  git clean -fdX  # (optional) after updating .gitignore")
endif()

# CUDA-Q Logical is always its own top-level CMake project. It is never added with
# add_subdirectory() from CUDA-Q: the only supported configuration builds
# against an *installed* CUDA-Q development SDK (see cmake/SetupCUDAQ.cmake).
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR
    "CUDA-Q Logical must be configured as a top-level project, not via add_subdirectory().\n"
    "Build it against an installed cudaq-devel SDK instead:\n"
    "  cmake -S preview/logical -B preview/logical/build")
endif()

project(cudaq-logical LANGUAGES CXX C)

set(QLX_USE_CUDAQ_SDK ON CACHE BOOL
  "Build CUDA-Q Logical against the installed CUDA-Q SDK")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(QLX_TOOLS_DIR "${CMAKE_BINARY_DIR}/bin"
  CACHE INTERNAL "Directory where CUDA-Q Logical command-line tools land")
set(QLX_PYTHON_DIR "${CMAKE_BINARY_DIR}/python"
  CACHE INTERNAL "Build-tree root of the CUDA-Q Logical Python package")

# Wheels keep native executables inside the package so their loader paths are
# stable relative to CUDA-Q's wheel-owned libraries.  Python console entry
# points provide the public commands in the environment's scripts directory.
# Preserve the conventional prefix/bin layout for ordinary CMake installs.
if(SKBUILD)
  set(QLX_TOOLS_INSTALL_DIR "cudaq/logical/_bin")
  if(APPLE)
    set(QLX_TOOLS_INSTALL_RPATH
      "@loader_path/../../../lib;@loader_path/../../../cuda_quantum.libs")
  else()
    set(QLX_TOOLS_INSTALL_RPATH
      "$ORIGIN/../../../lib;$ORIGIN/../../../cuda_quantum.libs")
  endif()
else()
  set(QLX_TOOLS_INSTALL_DIR "bin")
  set(QLX_TOOLS_INSTALL_RPATH "")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
option(QLX_BUILD_USE_CCACHE "Use ccache as the C/C++/CUDA compiler launcher" OFF)

# Use ccache as the C/C++/CUDA compiler launcher if it is available.
find_program(CCACHE_EXECUTABLE ccache)
if(CCACHE_EXECUTABLE)
  execute_process(
    COMMAND "${CCACHE_EXECUTABLE}" --version
    OUTPUT_VARIABLE _qlx_ccache_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" _qlx_ccache_version
    "${_qlx_ccache_version}")
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE STRING "" FORCE)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE STRING "" FORCE)
  # CUDA support added in ccache 3.7; nvcc routing is automatic in 4.x.
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_EXECUTABLE}" CACHE STRING "" FORCE)
  message(STATUS "Detected ccache:          ${CCACHE_EXECUTABLE} (${_qlx_ccache_version})")
endif()

# --------------------------------------------------------------------------- #
# Default build type
# --------------------------------------------------------------------------- #
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
  message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
endif()

# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #

# Resolve the Python environment first: SetupCUDAQ discovers the cudaq-devel
# wheel through it, and python/CMakeLists.txt has to see the same interpreter.
# Python + nanobind are unconditionally required — the Python surface is the
# canonical user entry point and a non-Python build is not supported.
include(QLXPythonEnv)
qlx_configure_python_env(REQUIRED)

include(SetupCUDAQ)
include(QLXTargetSetup)

# --------------------------------------------------------------------------- #
# Project include directories
# --------------------------------------------------------------------------- #

# TableGen-generated headers.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ir/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/ir/include)

# --------------------------------------------------------------------------- #
# Sub-directories
# --------------------------------------------------------------------------- #
add_subdirectory(ir)

# --------------------------------------------------------------------------- #
# Build-tree package config — lets external backend worktrees find CUDA-Q Logical via
# cmake -DQLX_DIR=<build>/lib/cmake/QLX ...
# --------------------------------------------------------------------------- #
export(TARGETS
  MLIRQLXDialect
  MLIRFabricDialect
  MLIRLVMDialect
  MLIRPhysDialect
  MLIRCflowDialect
  MLIREventDialect
  NAMESPACE QLX::
  FILE "${CMAKE_BINARY_DIR}/lib/cmake/QLX/QLXTargets.cmake"
)

configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/QLXConfig.cmake.in"
  "${CMAKE_BINARY_DIR}/lib/cmake/QLX/QLXConfig.cmake"
  @ONLY
)

enable_testing()

add_subdirectory(test)
add_subdirectory(python)
