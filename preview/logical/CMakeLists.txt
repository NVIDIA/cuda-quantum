# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.30)

# CUDA-Q Logical builds in one of two modes:
#
#  * In-tree: added as a subdirectory of the CUDA-Q build, either with
#    -DCUDAQ_ENABLE_LOGICAL=ON or through CUDAQ_EXTERNAL_PROJECTS. CUDA-Q's own
#    targets (cudaq::MLIR, the Python bindings, the LLVM/MLIR helpers) are used
#    directly, so the build is self-consistent by construction.
#  * Standalone: its own top-level project, built against an *installed* CUDA-Q
#    development SDK (see cmake/SetupCUDAQ.cmake). This is how the wheel builds.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(QLX_IN_TREE OFF)
else()
  set(QLX_IN_TREE ON)
endif()

if(NOT QLX_IN_TREE AND CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not allowed.\n"
    "  Source dir: ${CMAKE_SOURCE_DIR}\n"
    "  Build  dir: ${CMAKE_BINARY_DIR}\n"
    "Run 'cmake -S . -B build [options]' from the repo root instead.\n"
    "If you have already run cmake here, remove the stray files first:\n"
    "  rm -rf CMakeCache.txt CMakeFiles/ _deps/ Makefile cmake_install.cmake\n"
    "  git clean -fdX  # (optional) after updating .gitignore")
endif()

# Nothing here uses C++ modules. With C++20 and CMake >= 3.30, CMake otherwise
# scans every source for module dependencies and requires clang-scan-deps, which
# toolchains without it (e.g. the CI dev image) fail on. CUDA-Q's top-level
# CMakeLists.txt does the same, so in-tree builds already inherit it.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

project(cudaq-logical LANGUAGES CXX C)

set(QLX_USE_CUDAQ_SDK ON CACHE BOOL
  "Build CUDA-Q Logical against the installed CUDA-Q SDK")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT QLX_IN_TREE)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endif()

# In-tree, the tools share CUDA-Q's bin/: its $ORIGIN-relative RPATHs (to
# lib/ and the Python packages) are only valid from there.
if(QLX_IN_TREE)
  set(QLX_TOOLS_DIR "${CMAKE_BINARY_DIR}/bin"
    CACHE INTERNAL "Directory where CUDA-Q Logical command-line tools land")
else()
  set(QLX_TOOLS_DIR "${PROJECT_BINARY_DIR}/bin"
    CACHE INTERNAL "Directory where CUDA-Q Logical command-line tools land")
endif()
# In-tree, the Python package dovetails into CUDA-Q's own build-tree package
# (cudaq/, cudaq/mlir/, cudaq/mlir/_mlir_libs/), exactly like an installation.
if(QLX_IN_TREE)
  set(QLX_PYTHON_DIR "${MLIR_BINARY_DIR}/python"
    CACHE INTERNAL "Build-tree root of the CUDA-Q Python package")
else()
  set(QLX_PYTHON_DIR "${PROJECT_BINARY_DIR}/python"
    CACHE INTERNAL "Build-tree root of the CUDA-Q Logical Python package")
endif()

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
# (CUDA-Q's own build already does so when it is the parent project.)
if(NOT QLX_IN_TREE)
  find_program(CCACHE_EXECUTABLE ccache)
endif()
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

# EDEM supplies Logical's model values and native producer interface. Prefer
# the enclosing build or an explicit CMake package; wheels expose that package
# through the Python interpreter selected above.
if(NOT TARGET cudaq::edem)
  find_package(cudaq-edem 0.1 CONFIG QUIET)
  if(NOT cudaq-edem_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c
        "import cudaq_edem; print(cudaq_edem.get_cmake_dir())"
      OUTPUT_VARIABLE _qlx_edem_cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
      RESULT_VARIABLE _qlx_edem_python_result)
    if(_qlx_edem_python_result EQUAL 0)
      list(APPEND CMAKE_PREFIX_PATH "${_qlx_edem_cmake_dir}")
    endif()
    find_package(cudaq-edem 0.1 CONFIG REQUIRED)
  endif()
endif()

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

if(NOT QLX_IN_TREE)
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
    FILE "${PROJECT_BINARY_DIR}/lib/cmake/QLX/QLXTargets.cmake"
  )

  configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/QLXConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/lib/cmake/QLX/QLXConfig.cmake"
    @ONLY
  )
endif()

if(NOT QLX_IN_TREE)
  enable_testing()
endif()

add_subdirectory(test)
add_subdirectory(python)
