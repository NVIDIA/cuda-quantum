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
set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.cudaq_realtime" CACHE STRING
    "Install path prefix, prepended onto install directories")

# Project setup
# ==============================================================================

# Check if core is built as a standalone project.
project(cudaq-realtime)
set(CUDAQ_REALTIME_STANDALONE_BUILD TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The following must go after `project(...)` 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

set(CUDAQ_REALTIME_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDAQ_REALTIME_INCLUDE_DIR ${CUDAQ_REALTIME_SOURCE_DIR}/include)

# Add cmake directory to module path for custom Find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
# ==============================================================================

option(CUDAQ_REALTIME_BUILD_TESTS
       "Generate build targets for the CUDAQ real-time unit tests" ON)
option(CUDAQ_REALTIME_BUILD_EXAMPLES
       "Generate build targets for the CUDAQ real-time example programs" ON)
option(CUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS
       "Build Hololink bridge/emulator/playback tools (requires hololink)."
       OFF)

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
    if (CUDAToolkit_VERSION VERSION_LESS 13.0)
      # Ampere, Hopper
      set(CUDA_TARGET_ARCHS  "80;90")
    else()
      # Ampere, Hopper, Blackwell
      set(CUDA_TARGET_ARCHS  "80;90;100")
    endif()
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

# Enable static linking of the C++ standard library to avoid dependency issues when distributing the library.
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libstdc++ -static-libgcc")

add_subdirectory(lib)

if (CUDAQ_REALTIME_BUILD_EXAMPLES)
  message(STATUS "RoCE/DOCA examples removed for RPC dispatch workflow.")
endif()

if (CUDAQ_REALTIME_BUILD_TESTS)
  add_custom_target(CudaqRealtimeUnitTests)
  include(CTest)

  add_custom_target(run_tests
    COMMAND ${CMAKE_COMMAND} -E env 
            PYTHONPATH="${CUDAQ_INSTALL_DIR}:${CMAKE_BINARY_DIR}/python"
            ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS CudaqRealtimeUnitTests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  add_subdirectory(unittests)
endif()

