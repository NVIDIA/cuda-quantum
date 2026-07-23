# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(FetchContent)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" CUDAQ_REALTIME_STANDALONE_BUILD)

if(NOT CUDAQ_REALTIME_STANDALONE_BUILD)
  # The parent CUDA-Q build enables CMAKE_COMPILE_WARNING_AS_ERROR globally
  # (rendered as `-Werror all-warnings` for nvcc), but the standalone realtime
  # build does not. Follow the standalone warning policy for realtime targets
  # in the combined build as well.
  set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
endif()

if(CUDAQ_REALTIME_STANDALONE_BUILD)
  # Set a default build type if none was specified. Must set this before
  # project().
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel")

  # Set a default install prefix if none was specified.
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.cudaq_realtime" CACHE STRING
      "Install path prefix, prepended onto install directories")
endif()

# Project setup
# ==============================================================================

if(CUDAQ_REALTIME_STANDALONE_BUILD)
  project(cudaq-realtime)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

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

set(_cudaq_realtime_build_tests_default ON)
if(NOT CUDAQ_REALTIME_STANDALONE_BUILD)
  set(_cudaq_realtime_build_tests_default OFF)
endif()

option(CUDAQ_REALTIME_BUILD_TESTS
       "Generate build targets for the CUDAQ real-time unit tests"
       ${_cudaq_realtime_build_tests_default})
option(CUDAQ_REALTIME_BUILD_EXAMPLES
       "Generate build targets for the CUDAQ real-time example programs" ON)
option(CUDAQ_REALTIME_ENABLE_HOLOLINK_TOOLS
       "Build Hololink bridge/emulator/playback tools (requires hololink)."
       OFF)

set(_host_compiler_opts_list "")
if(CUDAQ_REALTIME_STANDALONE_BUILD)
  list(APPEND _host_compiler_opts_list "-fPIC")
endif()

list(JOIN _host_compiler_opts_list "," _host_compiler_opts)

# Check for CUDA Support (ref: cuda-quantum/CMakeLists.txt)
# ==============================================================================
if(CUDAQ_REALTIME_STANDALONE_BUILD)
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
    find_package(CUDAToolkit REQUIRED)
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
    # Keep realtime CUDA sources on C++17. nvcc's C++20 frontend can hit an
    # internal compiler error in libstdc++ headers when compiling the standalone
    # realtime tests, including test_host_dispatcher.cu, on the CI toolchain.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -shared -std=c++17 ${CUDA_gencode_flags} --compiler-options ${_host_compiler_opts}")

    enable_language(CUDA)
    set(CUDA_FOUND TRUE)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    message(STATUS "Cuda language found.")
  endif()
elseif(CUDA_FOUND)
  find_package(CUDAToolkit REQUIRED)
  if(_host_compiler_opts)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=${CMAKE_CXX_COMPILER} --compiler-options ${_host_compiler_opts}")
  endif()
endif()

# External Dependencies 
# ==============================================================================

find_package(Threads REQUIRED)

if(CUDAQ_REALTIME_STANDALONE_BUILD)
  # Standalone realtime owns its distribution linker policy. Integrated CUDA-Q
  # builds inherit the top-level CUDA-Q runtime/dependency link settings.
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
endif()

add_subdirectory(lib)

# Install the export set and package config unconditionally: members of
# cudaq-realtime-targets exist on CUDA-less configures too (the UDP transport
# is always built), and CMake silently installs an orphaned archive if a
# target joins an export set that is never itself installed. The config file
# only requires CUDAToolkit when this build had CUDA (see the @CUDA_FOUND@
# substitution in cudaq-realtime-config.cmake.in).
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cudaq-realtime-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/cudaq-realtime-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudaq-realtime)

install(EXPORT cudaq-realtime-targets
  FILE cudaq-realtime-targets.cmake
  NAMESPACE cudaq::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudaq-realtime
  COMPONENT Development)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/cudaq-realtime-config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudaq-realtime
  COMPONENT Development)

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


# Install the LICENSE and NOTICE files
install(FILES LICENSE NOTICE DESTINATION ${CMAKE_INSTALL_PREFIX})
