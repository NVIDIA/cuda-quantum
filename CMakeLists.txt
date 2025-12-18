# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

# Policies
# ==============================================================================
if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# CMP0135: Timestamp for content downloaded with ExternalProject_Add()
# New in CMake 3.24 https://cmake.org/cmake/help/latest/policy/CMP0135.html
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# Project setup
# ==============================================================================
project(cudaq LANGUAGES CXX C)

# Prevent In-source builds
# ==============================================================================
# Using the undocumented `CMAKE_DISABLE_IN_SOURCE_BUILDS` and
# `CMAKE_DISABLE_SOURCE_CHANCES` variables is not a good idea:  They can change
# without warning and they do not accomplish what they are supposed to do, i.e.,
# cmake files will still be created in the source tree.

# Going the extra mile to prevent the user from playing tricks with symlinks.
get_filename_component(REAL_PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}" REALPATH)
get_filename_component(REAL_PROJECT_BINARY_DIR "${PROJECT_BINARY_DIR}" REALPATH)

if("${REAL_PROJECT_SOURCE_DIR}" STREQUAL "${REAL_PROJECT_BINARY_DIR}")
  message(FATAL_ERROR
    "In-source builds are not permitted. You must run cmake in a separeted "
    "directory, e.g.:\n"
    "    mkdir build && cd build && cmake ..\n"
    "NOTE: Remeber to clean up the source tree by deleting the files already "
    "created by CMake, e.g, CMakeCache.txt and cmake.check_cache")
endif()

# Project globals
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Set warnings as errors by default. 
# Individual targets and the command line invocation can override this behavior.
if(NOT DEFINED CMAKE_COMPILE_WARNING_AS_ERROR)
  set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
endif()

# Enable the REST code by default.
if (NOT DEFINED CUDAQ_ENABLE_REST)
  set(CUDAQ_ENABLE_REST ON CACHE BOOL "Enable building REST client & server.")
endif()

# Enable the remote simulator by default.
if (CUDAQ_ENABLE_REST AND NOT DEFINED CUDAQ_ENABLE_REMOTE_SIM)
  set(CUDAQ_ENABLE_REMOTE_SIM ON CACHE BOOL "Enable building cudaq-qpud.")
  # Optionally enable the tests that use cudaq-qpud.
  if (NOT DEFINED CUDAQ_TEST_REMOTE_SIM)
    set(CUDAQ_TEST_REMOTE_SIM ON CACHE BOOL "Run remote-sim tests.")
  endif()
endif()

# Enable Amazon Braket backends by default.
if (NOT DEFINED CUDAQ_ENABLE_BRAKET_BACKEND)
  set(CUDAQ_ENABLE_BRAKET_BACKEND ON CACHE BOOL "Enable building AWS SDK for Amazon Braket backends.")
endif()

# Enable Pasqal target by default.
if (NOT DEFINED CUDAQ_ENABLE_PASQAL_BACKEND)
  set(CUDAQ_ENABLE_PASQAL_BACKEND ON CACHE BOOL "Enable building the Pasqal target.")
endif()

# Enable Quantum Circuits, Inc. (QCI) target by default.
if (NOT DEFINED CUDAQ_ENABLE_QCI_BACKEND)
  set(CUDAQ_ENABLE_QCI_BACKEND ON CACHE BOOL "Enable building the Quantum Circuits, Inc. target.")
endif()

# Enable Quantum Machines target by default.
if (NOT DEFINED CUDAQ_ENABLE_QUANTUM_MACHINES_BACKEND)
  set(CUDAQ_ENABLE_QUANTUM_MACHINES_BACKEND ON CACHE BOOL "Enable building the Quantum Machines target.")
endif()

# Generate a CompilationDatabase (compile_commands.json file) for our build,
# for use by clang_complete, YouCompleteMe, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if(NOT LLVM_VERSION_MAJOR)
  set(LLVM_VERSION_MAJOR 16)
endif()

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} -c submodule.tpls/llvm.update=none submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "Cloning git submodules failed with ${GIT_SUBMOD_RESULT}, please checkout submodules manually")
    endif()
  endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/tpls/fmt/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# Options
# ==============================================================================
option(CUDAQ_BUILD_TESTS "Build cudaq tests" ON)
option(CUDAQ_ENABLE_RPC_LOGGING "Enable verbose printout for client/server qpud connection." OFF)
option(CUDAQ_TEST_MOCK_SERVERS "Enable Remote QPU Tests via Mock Servers." OFF)
option(CUDAQ_DISABLE_RUNTIME "Build without the CUDA-Q runtime, just the compiler toolchain." OFF)
option(CUDAQ_SKIP_MPI "Do not build with MPI, even when it is available." OFF)
option(CUDAQ_DISABLE_CPP_FRONTEND "Build without the CUDA-Q C++ Clang-based Frontend." OFF)
option(CUDAQ_ENABLE_CC "Enable CUDA-Q code coverage generation." OFF)
option(CUDAQ_REQUIRE_OPENMP "Fail the build if OpenMP is not found." OFF)

# Certain build configurations may be set directly in the environment.
# This facilitates some of the packaging (e.g. python packages built based on the pyproject.toml).
if(NOT LLVM_DIR AND EXISTS "$ENV{LLVM_INSTALL_PREFIX}/lib/cmake/llvm")
  SET(LLVM_DIR "$ENV{LLVM_INSTALL_PREFIX}/lib/cmake/llvm")
endif()
if(NOT BLAS_LIBRARIES AND EXISTS "$ENV{BLAS_INSTALL_PREFIX}/libblas.a")
  # CACHE INTERNAL is needed due to how FindBLAS.cmake works...
  SET(BLAS_LIBRARIES "$ENV{BLAS_INSTALL_PREFIX}/libblas.a" CACHE INTERNAL "")	
endif()
if(NOT CUSTATEVEC_ROOT)
  SET(CUSTATEVEC_ROOT "$ENV{CUQUANTUM_INSTALL_PREFIX}")
endif()
if(NOT CUTENSORNET_ROOT)
  SET(CUTENSORNET_ROOT "$ENV{CUQUANTUM_INSTALL_PREFIX}")
endif()
if(NOT CUDENSITYMAT_ROOT)
  SET(CUDENSITYMAT_ROOT "$ENV{CUQUANTUM_INSTALL_PREFIX}")
endif()
if(NOT CUTENSOR_ROOT)
  SET(CUTENSOR_ROOT "$ENV{CUTENSOR_INSTALL_PREFIX}")
endif()
if(NOT ZLIB_ROOT)
  SET(ZLIB_ROOT "$ENV{ZLIB_INSTALL_PREFIX}")	
endif()
if(NOT OPENSSL_ROOT_DIR)	
  SET(OPENSSL_ROOT_DIR "$ENV{OPENSSL_INSTALL_PREFIX}")	
endif()
if (NOT crypto_LIBRARY AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" AND EXISTS "$ENV{OPENSSL_INSTALL_PREFIX}/lib64/libcrypto.a")
  SET(crypto_LIBRARY "$ENV{OPENSSL_INSTALL_PREFIX}/lib64/libcrypto.a" CACHE INTERNAL "")
elseif(NOT crypto_LIBRARY AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" AND EXISTS "$ENV{OPENSSL_INSTALL_PREFIX}/lib/libcrypto.a")
  SET(crypto_LIBRARY "$ENV{OPENSSL_INSTALL_PREFIX}/lib/libcrypto.a" CACHE INTERNAL "")
endif()
if (NOT crypto_INCLUDE_DIR AND EXISTS "$ENV{OPENSSL_INSTALL_PREFIX}/include")
  SET(crypto_INCLUDE_DIR "$ENV{OPENSSL_INSTALL_PREFIX}/include" CACHE INTERNAL "")
endif()
if(NOT CURL_LIBRARY AND EXISTS "$ENV{CURL_INSTALL_PREFIX}/lib/libcurl.a")
  SET(CURL_LIBRARY "$ENV{CURL_INSTALL_PREFIX}/lib/libcurl.a")	
  SET(CURL_INCLUDE_DIR "$ENV{CURL_INSTALL_PREFIX}/include")
  SET(CURL_CONFIG_EXECUTABLE "$ENV{CURL_INSTALL_PREFIX}/bin/curl-config")
  SET(CMAKE_USE_SYSTEM_CURL TRUE)
  SET(CURL_NO_CURL_CMAKE ON)
endif()
if(NOT AWSSDK_ROOT AND CUDAQ_ENABLE_BRAKET_BACKEND)
  SET(AWSSDK_ROOT "$ENV{AWS_INSTALL_PREFIX}")
endif()
if(NOT CUDAQ_EXTERNAL_NVQIR_SIMS)
  SET(CUDAQ_EXTERNAL_NVQIR_SIMS $ENV{CUDAQ_EXTERNAL_NVQIR_SIMS})
endif()
if(NOT CUDAQ_ENABLE_CC)
  SET(CUDAQ_ENABLE_CC $ENV{CUDAQ_ENABLE_CC})
endif()

SET(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_SKIP_INSTALL_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
SET(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/lib/plugins:$ORIGIN/../lib:$ORIGIN/../lib/plugins")

SET(BLA_STATIC ON)
if(NOT DEFINED ZLIB_USE_STATIC_LIBS)
  SET(ZLIB_USE_STATIC_LIBS TRUE)
endif()
if(NOT DEFINED OPENSSL_USE_STATIC_LIBS)
  SET(OPENSSL_USE_STATIC_LIBS TRUE)
endif()

if(NOT DEFINED CUDAQ_ENABLE_STATIC_LINKING)
  SET(CUDAQ_ENABLE_STATIC_LINKING $ENV{CUDAQ_ENABLE_STATIC_LINKING})
endif()
if (CUDAQ_ENABLE_STATIC_LINKING)
  # Using -static-libgcc -static-libstdc++ offers more fine grained control here
  # then using -static (for exe) and -staic-pie (for linker).
  # At the time of writing, there wasn't any benefit to using those over -static.
  # When we only link libstdc++ and libgcc statically, we still have a dependency
  # on GLIBC on the target system, but linking it statically is likely to cause issues. 
  # A better option might be to include the required GNU C Library and have the nvq++
  # compiler dynamically pick the newer one of the included or system one.
  # See also these posts:
  # - https://blog.gibson.sh/2017/11/26/creating-portable-linux-binaries/
  # - https://www.jertype.com/upgrading-glibc/
  # If the final application loads multiple dynamic libraries, and two or more
  # of those dynamic libraries are linked to different versions of the same static
  # libraries, then the different copies of those static libraries will conflict with
  # each other. If we were to link everything statically, it is probably advisable to 
  # set CMAKE_<LANG>_VISIBILITY_PRESET to hidden and CMAKE_VISIBILITY_INLINES_HIDDEN 
  # to 1, to avoid some of these issues.
  # We don't set these configurations here, since the LLVM runtime libraries that we 
  # build from source are already configured to hide symbols and include dependencies.
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
  SET(OPENSSL_USE_STATIC_LIBS TRUE)
  SET(ZLIB_USE_STATIC_LIBS TRUE)
  set(CURL_STATICLIB ON)
  message(STATUS "CUDA Quantum static linking enabled.")
else()
  message(STATUS "CUDA Quantum static linking disabled.")
endif()

# ZLIB is required when we build LLVM with zlib support. 
# ZLIB support in LLVM is good to have e.g. for the linker,
# since some binaries will use zlib compression to store symbols.
find_package(ZLIB REQUIRED)
include(ExternalProject)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif()

# Find and configure LLVM, Clang and MLIR
# ==============================================================================
# The user can specify the path to LLVM cmake directory using
# `-DLLVM_DIR=path/to/cmake/llvm`.  If this definition is not provided, we look
# for `llvm-config` tool.  The user can also provide a LLVM version to look for
# using `LLVM_VERSION_MAJOR`, e.g. "-LLVM_VERSION_MAJOR=16".  Note that this
# version variable is set to the latest LLVM version by default, and setting it
# to an older version might break the project.
find_package(LLVM ${LLVM_VERSION_MAJOR} CONFIG QUIET)

if(NOT LLVM_DIR)
  message(STATUS "LLVM_DIR not found, will try with llvm-config executable.")

  macro(find_llvm_config name version_major)
    set(extra_args ${ARGN})
    list(LENGTH extra_args extra_count)
    if (${extra_count} GREATER 0)
      list(GET extra_args 0 path)
      find_program(LLVM_CONFIG NAMES ${name} PATHS ${path} NO_DEFAULT_PATH
        DOC "Path to llvm-config tool")
    else()
      find_program(LLVM_CONFIG NAMES ${name} DOC "Path to llvm-config tool")
    endif()

    if(LLVM_CONFIG)
      execute_process(
        COMMAND ${LLVM_CONFIG} --version
        RESULT_VARIABLE LLVM_CONFIG_RESULT
        OUTPUT_VARIABLE LLVM_CONFIG_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )

      if(NOT LLVM_CONFIG_VERSION MATCHES "^${version_major}[.][0-9]+[.][0-9]+")
        unset(LLVM_CONFIG CACHE)
      endif()
    endif()
  endmacro()

  # First we try to find llvm-config in the llvm submodule.
  find_llvm_config(llvm-config ${LLVM_VERSION_MAJOR}
    "${CMAKE_CURRENT_SOURCE_DIR}/tpls/llvm/build/bin")

  # Try to find a system llvm-config and make sure it is the correct version.
  if(NOT LLVM_CONFIG)
    find_llvm_config(llvm-config ${LLVM_VERSION_MAJOR})
  endif()

  # If it is not the correct version, try finding llvm-config-VERSION
  if(NOT LLVM_CONFIG)
    find_llvm_config(llvm-config-${LLVM_VERSION_MAJOR} ${LLVM_VERSION_MAJOR})
    if (LLVM_CONFIG)
      set(NVQPP_LLVM_EXECUTABLE_SUFFIX -${LLVM_VERSION_MAJOR})
    endif()
  endif()

  execute_process(
    COMMAND ${LLVM_CONFIG} --cmakedir
    RESULT_VARIABLE LLVM_CONFIG_RESULT
    OUTPUT_VARIABLE LLVM_CONFIG_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  if(LLVM_CONFIG_RESULT)
    message(FATAL_ERROR
      "Could not find suitable llvm-config(-${LLVM_VERSION_MAJOR}).\
      \nTry providing valid -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm.")
  else()
    find_package(LLVM ${LLVM_VERSION_MAJOR} REQUIRED CONFIG
      HINTS ${LLVM_CONFIG_CMAKE_DIR} NO_DEFAULT_PATH)
  endif()
endif()

if(NOT MLIR_DIR)
  set(MLIR_DIR ${LLVM_BINARY_DIR}/lib/cmake/mlir)
endif()

find_package(MLIR REQUIRED CONFIG)

if(NOT Clang_DIR)
  set(Clang_DIR ${LLVM_BINARY_DIR}/lib/cmake/clang)
endif()

find_package(Clang CONFIG)
if (NOT Clang_FOUND) 
  message(STATUS "Clang not found, turning off C++ Frontend.")
  set (CUDAQ_DISABLE_CPP_FRONTEND ON)
endif()

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Add LLVM, Clang and MLIR headers to the include path
include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS})

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")

if(MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()

set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

# CUDA Quantum configuration
# ==============================================================================
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

set(CUDAQ_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}) # --src-root
set(CUDAQ_MAIN_INCLUDE_DIR ${CUDAQ_MAIN_SRC_DIR}/include)

set(CUDAQ_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDAQ_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CUDAQ_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CUDAQ_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(CUDAQ_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
set(CUDAQ_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${CUDAQ_CMAKE_DIR}")

include(AddCUDAQ)
include(BuildHelpers)

# Add CUDA Quantum files to the include path
include_directories(BEFORE
  ${CUDAQ_SOURCE_DIR}/include
  ${CUDAQ_BINARY_DIR}/include)

# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
add_custom_target(cudaq-headers)
set_target_properties(cudaq-headers PROPERTIES FOLDER "Misc")
add_custom_target(cudaq-doc)

# Version
# ==============================================================================
if (DEFINED ENV{CUDA_QUANTUM_VERSION})
  # The version was defined by the user (likely a bot performing the build), so
  # use the value provided as is.
  set(CUDA_QUANTUM_VERSION "$ENV{CUDA_QUANTUM_VERSION}")
else()
  # Otherwise, create a version based on the nearest tag in the git repo.
  execute_process(COMMAND git describe --tags --abbrev=0 --dirty=-developer
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE CUDA_QUANTUM_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# Retrieve the commit SHA for full revision description
execute_process(COMMAND git rev-parse --verify HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE CUDA_QUANTUM_COMMIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE)

configure_file("${CMAKE_SOURCE_DIR}/include/cudaq/Support/Version.h.in" 
               "${CUDAQ_BINARY_DIR}/include/cudaq/Support/Version.h" @ONLY)

# Check optional dependencies
# ==============================================================================

find_package(OpenSSL)
if (NOT CUDAQ_SKIP_MPI)
  find_package(MPI COMPONENTS CXX)
  if (MPI_FOUND)
    message(STATUS "MPI CXX Found: ${MPIEXEC}")
    # Build the built-in MPI Comm plugin
    add_subdirectory(runtime/cudaq/distributed/builtin)
  endif()
endif()

# Third-party libraries (tpls)
# ==============================================================================
set(FMT_INSTALL ON)
cmake_policy(PUSH)
# CMP0146: The FindCUDA module is removed.
# CMake 3.27 and above no longer provide the modules.
# https://cmake.org/cmake/help/latest/policy/CMP0146.html
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
add_subdirectory(tpls/fmt)
cmake_policy(POP)
include_directories(SYSTEM tpls/json/include)

# Add spdlog 
set(SPDLOG_BUILD_SHARED OFF)
set(SPDLOG_FMT_EXTERNAL ON)
set(SPDLOG_BUILD_PIC ON)
set(SPDLOG_SYSTEM_INCLUDES ON)
add_subdirectory(tpls/spdlog)
target_compile_options(spdlog PRIVATE -Wno-covered-switch-default)

if (OPENSSL_FOUND AND CUDAQ_ENABLE_REST)
  # Look for CURL
  set(CURL_STATICLIB ON)
  set(CURL_USE_STATIC_LIBS TRUE)
  find_package(CURL COMPONENTS HTTP HTTPS)

  if (CURL_FOUND)
    message(STATUS "Using system curl for CPR.")
    set(CPR_FORCE_USE_SYSTEM_CURL TRUE)
  elseif (CMAKE_USE_SYSTEM_CURL)
    message(FATAL_ERROR "CMAKE_USE_SYSTEM_CURL defined but a suitable package was not found. Make sure that `curl-config --protocols` lists HTTP and HTTPS.")
  else()
    message(STATUS "Curl not found. Building curl for CPR.")
    set(CPR_FORCE_USE_SYSTEM_CURL FALSE)
  endif()

  # Now, build CPR as needed
  set(BUILD_SHARED_LIBS OFF)
  set(CPR_ENABLE_SSL ON) # needed for https requests
  set(CPR_FORCE_OPENSSL_BACKEND ON)
  add_subdirectory(tpls/cpr)
  target_compile_options(cpr PRIVATE -Wno-covered-switch-default -w)
endif()

# QPP simulator
if (NOT CUDAQ_DISABLE_RUNTIME)
  add_subdirectory(tpls/qpp EXCLUDE_FROM_ALL)
  # Required workaround to add libqpp as SYSTEM. This will prevent it from
  # generating warnings, which we treat as errors.
  get_target_property(LIBQPP_INCLUDE libqpp INTERFACE_INCLUDE_DIRECTORIES)
  set_target_properties(libqpp
    PROPERTIES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${LIBQPP_INCLUDE}")
  add_openmp_interface_definitions(libqpp)
endif()

if (OPENSSL_FOUND AND CUDAQ_ENABLE_REST)
  # The asio submodule doesn't use cmake, so use find_path for it.
  find_path(ASIO_INCLUDE_DIR asio.hpp PATHS tpls/asio/asio/include)
  add_subdirectory(tpls/Crow)
endif()

# Check for CUDA Support
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
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "Found CUDA Toolkit version: ${CUDAToolkit_VERSION}")

    if (CUDAToolkit_VERSION VERSION_LESS 13.0)
      # Turing, Ampere, Hopper
      set(CUDA_TARGET_ARCHS  "75;80;90")
    else()
      # Turing, Ampere, Hopper, Blackwell
      set(CUDA_TARGET_ARCHS  "75;80;90;100")
    endif()
  endif()
  CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_TARGET_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -shared -std=c++20 ${CUDA_gencode_flags} --compiler-options -fPIC")

  enable_language(CUDA)
  set(CUDA_FOUND TRUE)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  message(STATUS "Cuda language found.")
endif()

# Code coverage setup
# ==============================================================================
if(CUDAQ_ENABLE_CC)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
  endif()
endif(CUDAQ_ENABLE_CC)

# Directory setup
# ==============================================================================

add_subdirectory(cmake)
add_subdirectory(include)
add_subdirectory(lib)
if (NOT CUDAQ_DISABLE_RUNTIME)
  add_subdirectory(runtime)
endif()
add_subdirectory(tools)
add_subdirectory(utils)

if(CUDAQ_BUILD_TESTS)
  enable_testing()
endif()

if (CUDAQ_ENABLE_PYTHON)
  find_package(Python 3 COMPONENTS Interpreter Development)
  
  # Apply specific patch to pybind11 for our documentation.
  # Only apply the patch if not already applied.
  execute_process(COMMAND ${GIT_EXECUTABLE} -C tpls/pybind11/ apply ../customizations/pybind11/pybind.h.diff --ignore-whitespace --reverse --check
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  RESULT_VARIABLE GIT_PATCH_RESULT
                  ERROR_QUIET)
  if (NOT GIT_PATCH_RESULT EQUAL "0")
    execute_process(COMMAND ${GIT_EXECUTABLE} -C tpls/pybind11/ apply ../customizations/pybind11/pybind.h.diff --ignore-whitespace
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_PATCH_RESULT)
  endif()
  if (NOT GIT_PATCH_RESULT EQUAL "0")
    message(FATAL_ERROR "Applying patch to submodule failed with ${GIT_PATCH_RESULT}, please update patch")
  endif()
  
  # Regarding the use of PyBind, we need to be careful that the same STL is used for any 
  # Python bindings generated as part of the CUDA-Q build and bindings generated for 
  # third party CUDA-Q libraries; see also https://github.com/pybind/pybind11/issues/1262
  add_subdirectory(tpls/pybind11)
  add_subdirectory(python)
endif()

if(CUDAQ_BUILD_TESTS AND NOT CUDAQ_DISABLE_CPP_FRONTEND)
  umbrella_lit_testsuite_begin(check-all)
  set(INSTALL_GTEST OFF)
  add_subdirectory(tpls/googletest-src)
  # Bug in GCC 12 leads to spurious warnings (-Wrestrict)
  # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
  if (CMAKE_COMPILER_IS_GNUCXX 
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0 
    AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
    target_compile_options(gtest PUBLIC --param=evrp-mode=legacy)
  endif()
  if (CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    target_compile_options(gtest PUBLIC -Wno-covered-switch-default)
  endif()
  include(GoogleTest)
  include(CTest)
  add_subdirectory(test)
  if (NOT CUDAQ_DISABLE_RUNTIME)
    add_subdirectory(targettests)
    add_subdirectory(unittests)
    add_subdirectory(docs)
  endif()
  umbrella_lit_testsuite_end(check-all)
endif()

if (CUDAQ_EXTERNAL_NVQIR_SIMS) 
  while(CUDAQ_EXTERNAL_NVQIR_SIMS)
    list(POP_FRONT CUDAQ_EXTERNAL_NVQIR_SIMS LIB_SO_OR_CONFIG_FILE)
    add_target_libs_to_wheel(${LIB_SO_OR_CONFIG_FILE})
  endwhile()
endif()
