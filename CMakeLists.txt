# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Requiring the same version as the others.
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

include(FetchContent)

# Set a default build type if none was specified. Must set this before
# project().
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel")

# Set a default install prefix if none was specified.
set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.cudaqx" CACHE STRING
    "Install path prefix, prepended onto install directories")

# Project setup
# ==============================================================================

# Check if core is built as a standalone project.
project(cudaq-nvqlink)
set(CUDAQ_NVQLINK_STANDALONE_BUILD TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The following must go after `project(...)` 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

set(CUDAQ_NVQLINK_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDAQ_NVQLINK_INCLUDE_DIR ${CUDAQ_NVQLINK_SOURCE_DIR}/include)

# Options
# ==============================================================================

option(NVQLINK_BUILD_TESTS
       "Generate build targets for the NVQLINK unit tests" ON)
option(NVQLINK_BUILD_EXAMPLES
       "Generate build targets for the NVQLINK example programs" ON)
option(NVQLINK_ENABLE_ROCE
       "Enable RoCE backend using libibverbs" OFF)

# Profiler backend selection
set(NVQLINK_PROFILER_BACKEND "NONE" CACHE STRING "Profiler backend (NONE, NVTX, TRACY)")
set_property(CACHE NVQLINK_PROFILER_BACKEND PROPERTY STRINGS NONE NVTX TRACY)

# Logging backend selection
set(NVQLINK_LOGGING_BACKEND "NONE" CACHE STRING "Logging backend (NONE, QUILL)")
set_property(CACHE NVQLINK_LOGGING_BACKEND PROPERTY STRINGS NONE QUILL)

# Compile-time log level filtering (lower levels become no-ops)
set(NVQLINK_LOGGING_LEVEL "INFO" CACHE STRING "Minimum log level (TRACE, DEBUG, INFO, WARNING, ERROR)")
set_property(CACHE NVQLINK_LOGGING_LEVEL PROPERTY STRINGS TRACE DEBUG INFO WARNING ERROR)

# Check for CUDA Support (ref: cuda-quantum/CMakeLists.txt)
# ==============================================================================
include(CheckLanguage)
check_language(CUDA)
set(CUDA_FOUND FALSE)
# Generate -gencode arch=compute_XX,code=sm_XX for list of supported
# arch values.
# List should be sorted in increasing order.
function(CUDA_get_gencode_args out_args_string arch_values)
  # allow the user to pass the list like a normal variable
  set(arch_list ${arch_values} ${ARGN})
  set(out "")
  foreach(arch IN LISTS arch_list)
    set(out "${out} -gencode arch=compute_${arch},code=sm_${arch}")
  endforeach(arch)

  # Repeat the last one as to ensure the generation of PTX for most
  # recent virtual architecture for forward compatibility
  list(GET arch_list -1 last_arch)
  set(out "${out} -gencode arch=compute_${last_arch},code=compute_${last_arch}")
  set(${out_args_string} ${out} PARENT_SCOPE)
endfunction()

if(CMAKE_CUDA_COMPILER)
  if (NOT CUDA_TARGET_ARCHS)
    # Ampere, Hopper
    set(CUDA_TARGET_ARCHS  "80;90")
  endif()
  CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_TARGET_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -shared -std=c++17 ${CUDA_gencode_flags} --compiler-options -fPIC")

  enable_language(CUDA)
  set(CUDA_FOUND TRUE)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Cuda language found.")
endif()

# External Dependencies 
# ==============================================================================

find_package(Threads REQUIRED)

add_subdirectory(lib)

if (NVQLINK_BUILD_EXAMPLES)
  add_subdirectory(examples/roce)
endif()

if (NVQLINK_BUILD_TESTS)
  add_custom_target(NVQLINKUnitTests)
  include(CTest)

  add_custom_target(run_tests
    COMMAND ${CMAKE_COMMAND} -E env 
            PYTHONPATH="${CUDAQ_INSTALL_DIR}:${CMAKE_BINARY_DIR}/python"
            ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS NVQLINKUnitTests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  add_subdirectory(unittests)
endif()

