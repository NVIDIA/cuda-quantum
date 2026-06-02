# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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
  cmake_policy(SET CMP0116 NEW)
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
    "In-source builds are not permitted. You must run cmake in a separated "
    "directory, e.g.:\n"
    "    mkdir build && cd build && cmake ..\n"
    "NOTE: Remember to clean up the source tree by deleting the files already "
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

# CUDAQ_ENABLE_ALL_BACKEND, if defined, becomes the default value used for
# every per-backend CUDAQ_ENABLE_<NAME>_BACKEND flag below. Individual
# per-backend flags can still be overridden on the command line to opt in
# or out of specific backends.
#
# Examples:
#   # Disable every backend:
#   cmake -DCUDAQ_ENABLE_ALL_BACKEND=OFF ..
#   # Disable every backend except IonQ:
#   cmake -DCUDAQ_ENABLE_ALL_BACKEND=OFF -DCUDAQ_ENABLE_IONQ_BACKEND=ON ..
if (DEFINED CUDAQ_ENABLE_ALL_BACKEND)
  set(_cudaq_backend_default ${CUDAQ_ENABLE_ALL_BACKEND})
else()
  set(_cudaq_backend_default ON)
endif()

# Enable Amazon Braket backends by default.
if (NOT DEFINED CUDAQ_ENABLE_BRAKET_BACKEND)
  set(CUDAQ_ENABLE_BRAKET_BACKEND ${_cudaq_backend_default} CACHE BOOL "Enable building AWS SDK for Amazon Braket backends.")
endif()

# Enable Anyon target by default.
if (NOT DEFINED CUDAQ_ENABLE_ANYON_BACKEND)
  set(CUDAQ_ENABLE_ANYON_BACKEND ${_cudaq_backend_default} CACHE BOOL "Enable building the Anyon target.")
endif()

# Enable IonQ target by default.
if (NOT DEFINED CUDAQ_ENABLE_IONQ_BACKEND)
  set(CUDAQ_ENABLE_IONQ_BACKEND ${_cudaq_backend_default} CACHE BOOL "Enable building the IonQ target.")
endif()

# Enable IQM target by default.
if (NOT DEFINED CUDAQ_ENABLE_IQM_BACKEND)
  set(CUDAQ_ENABLE_IQM_BACKEND ${_cudaq_backend_default} CACHE BOOL "Enable building the IQM target.")
endif()

# Enable OQC target by default.
if (NOT DEFINED CUDAQ_ENABLE_OQC_BACKEND)
  set(CUDAQ_ENABLE_OQC_BACKEND ${_cudaq_backend_default} CACHE BOOL "Enable building the OQC target.")
endif()

# Enable Pasqal target by default.
if (NOT DEFINED CUDAQ_ENABLE_PASQAL_BACKEND)
  set(CUDAQ_ENABLE_PASQAL_BACKEND ${_cudaq_backend_default} CACHE BOOL "Enable building the Pasqal target.")
endif()

# Enable Pasqal QRMI connector by default only on Linux builds, and only when
# the Pasqal backend itself is enabled. Both behaviors can be overridden by
# explicitly passing -DCUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR=ON/OFF on the cmake
# command line.
if (NOT DEFINED CUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR)
  if (NOT CUDAQ_ENABLE_PASQAL_BACKEND)
    set(CUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR OFF CACHE BOOL "Enable building the Pasqal QRMI connector.")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR ON CACHE BOOL "Enable building the Pasqal QRMI connector.")
  else()
    set(CUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR OFF CACHE BOOL "Enable building the Pasqal QRMI connector.")
  endif()
elseif (CUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(WARNING "CUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR=ON is supported only on Linux; forcing OFF.")
  set(CUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR OFF CACHE BOOL "Enable building the Pasqal QRMI connector." FORCE)
endif()

# Enable Quantum Circuits, Inc. (QCI) target by default.
if (NOT DEFINED CUDAQ_ENABLE_QCI_BACKEND)
  set(CUDAQ_ENABLE_QCI_BACKEND ${_cudaq_backend_default} CACHE BOOL "Enable building the Quantum Circuits, Inc. target.")
endif()

# Enable TII target by default.
if (NOT DEFINED CUDAQ_ENABLE_TII_BACKEND)
  set(CUDAQ_ENABLE_TII_BACKEND ${_cudaq_backend_default} CACHE BOOL "Enable building the TII target.")
endif()

# Enable Quantum Machines target by default.
if (NOT DEFINED CUDAQ_ENABLE_QUANTUM_MACHINES_BACKEND)
  set(CUDAQ_ENABLE_QUANTUM_MACHINES_BACKEND ${_cudaq_backend_default} CACHE BOOL "Enable building the Quantum Machines target.")
endif()

# Enable Scaleway target by default.
if (NOT DEFINED CUDAQ_ENABLE_SCALEWAY_BACKEND)
  set(CUDAQ_ENABLE_SCALEWAY_BACKEND ${_cudaq_backend_default} CACHE BOOL "Enable building the Scaleway target.")
endif()

# Enable qBraid target by default.
if (NOT DEFINED CUDAQ_ENABLE_QBRAID_BACKEND)
  set(CUDAQ_ENABLE_QBRAID_BACKEND ${_cudaq_backend_default} CACHE BOOL "Enable building the qBraid target.")
endif()

# =============================================================================
# CUDAQ_EXTERNAL_PROJECTS
# -----------------------------------------------------------------------------
# Plug additional sub-projects into the CUDA-Q build, mirroring LLVM's
# LLVM_EXTERNAL_PROJECTS pattern.
#
# Inputs
#   CUDAQ_EXTERNAL_PROJECTS
#       Semicolon-separated list of project names to include in the build.
#       Each name is a free-form identifier; dashes are allowed.
#
#   CUDAQ_EXTERNAL_<UPPER_NAME>_SOURCE_DIR
#       Absolute path to the source directory containing the project's
#       CMakeLists.txt. Required for every name listed in
#       CUDAQ_EXTERNAL_PROJECTS. <UPPER_NAME> is the project name uppercased
#       with dashes converted to underscores (e.g. project "my-tool"
#       -> CUDAQ_EXTERNAL_MY_TOOL_SOURCE_DIR).
#
# Build layout
#   Each external project is added via add_subdirectory() after the in-tree
#   sub-projects (cudaq/, runtime/, python/), so externals may depend on
#   CUDA-Q targets. The binary directory for each project is placed at
#   ${CMAKE_BINARY_DIR}/external/<name>.
#
# Diagnostics
#   * Listing a project without setting its SOURCE_DIR is a fatal error.
#   * A SOURCE_DIR that does not contain a CMakeLists.txt is a fatal error.
#
# Examples
#   # Single external project:
#   cmake -DCUDAQ_EXTERNAL_PROJECTS="my-tool" \
#         -DCUDAQ_EXTERNAL_MY_TOOL_SOURCE_DIR=/path/to/my-tool ..
#
#   # Multiple external projects:
#   cmake -DCUDAQ_EXTERNAL_PROJECTS="foo;bar" \
#         -DCUDAQ_EXTERNAL_FOO_SOURCE_DIR=/path/to/foo \
#         -DCUDAQ_EXTERNAL_BAR_SOURCE_DIR=/path/to/bar ..
# =============================================================================
set(CUDAQ_EXTERNAL_PROJECTS "" CACHE STRING
    "Semicolon-separated list of external projects to add to the CUDA-Q build.")
foreach(_cudaq_ext_proj IN LISTS CUDAQ_EXTERNAL_PROJECTS)
  string(TOUPPER "${_cudaq_ext_proj}" _cudaq_ext_upper)
  string(REGEX REPLACE "-" "_" _cudaq_ext_upper "${_cudaq_ext_upper}")
  set(CUDAQ_EXTERNAL_${_cudaq_ext_upper}_SOURCE_DIR "" CACHE PATH
      "Source directory for external project '${_cudaq_ext_proj}'.")
endforeach()

# Generate a CompilationDatabase (compile_commands.json file) for our build,
# for use by clang_complete, YouCompleteMe, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if(NOT LLVM_VERSION_MAJOR)
  set(LLVM_VERSION_MAJOR 22)
endif()
if(NOT LLVM_VERSION_MINOR)
  set(LLVM_VERSION_MINOR 1)
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
option(CUDAQ_ENABLE_SANITIZERS "Enable Address Sanitizer (ASan) and Undefined Behavior Sanitizer (UBSan)." OFF)
option(CUDAQ_ENABLE_CLIFFORD_T_SYNTHESIS "Enable the Clifford+T rotation synthesis library (requires GMP and MPFR; the build fails if those are missing)." OFF)

# Sub-projects that can be enabled independently, in the style of
# LLVM_ENABLE_PROJECTS. Accepts a semicolon-separated list.
set(CUDAQ_ALL_PROJECTS "cudaq;runtime;python")
set(CUDAQ_ENABLE_PROJECTS "" CACHE STRING
    "Semicolon-separated list of CUDA-Q sub-projects to build. Valid values: ${CUDAQ_ALL_PROJECTS}. Defaults to 'cudaq;runtime' (plus 'python' if CUDAQ_ENABLE_PYTHON is set).")

# Back-compat: if CUDAQ_ENABLE_PROJECTS wasn't given, derive it from the
# legacy per-project flags so existing build scripts keep working, and warn
# the user that those flags are deprecated.
if(NOT CUDAQ_ENABLE_PROJECTS)
  set(CUDAQ_ENABLE_PROJECTS "cudaq")
  if(NOT CUDAQ_DISABLE_RUNTIME)
    list(APPEND CUDAQ_ENABLE_PROJECTS "runtime")
  endif()
  if(CUDAQ_ENABLE_PYTHON)
    list(APPEND CUDAQ_ENABLE_PROJECTS "python")
  endif()

  if(CUDAQ_DISABLE_RUNTIME)
    message(DEPRECATION
      "CUDAQ_DISABLE_RUNTIME is deprecated. Use CUDAQ_ENABLE_PROJECTS to "
      "control which sub-projects are built (omit 'runtime' to disable it).")
  endif()
  if(DEFINED CUDAQ_ENABLE_PYTHON)
    message(DEPRECATION
      "CUDAQ_ENABLE_PYTHON is deprecated. Use CUDAQ_ENABLE_PROJECTS to "
      "control which sub-projects are built (include 'python' to enable it).")
  endif()
endif()

foreach(_proj IN LISTS CUDAQ_ENABLE_PROJECTS)
  if(NOT _proj IN_LIST CUDAQ_ALL_PROJECTS)
    message(FATAL_ERROR
      "Unknown project '${_proj}' in CUDAQ_ENABLE_PROJECTS. "
      "Valid: ${CUDAQ_ALL_PROJECTS}.")
  endif()
endforeach()

# python depends on the MLIR dialects in cudaq/ and on runtime symbols.
if("python" IN_LIST CUDAQ_ENABLE_PROJECTS)
  foreach(_dep cudaq runtime)
    if(NOT _dep IN_LIST CUDAQ_ENABLE_PROJECTS)
      message(STATUS "CUDAQ_ENABLE_PROJECTS: adding implied dep '${_dep}' (required by python).")
      list(APPEND CUDAQ_ENABLE_PROJECTS ${_dep})
    endif()
  endforeach()
endif()

# Map CUDAQ_ENABLE_PROJECTS onto the legacy per-project flags so the rest of
# this file and sub-CMakes keep working unchanged.
if(NOT "runtime" IN_LIST CUDAQ_ENABLE_PROJECTS)
  set(CUDAQ_DISABLE_RUNTIME ON CACHE BOOL "" FORCE)
endif()
if("python" IN_LIST CUDAQ_ENABLE_PROJECTS)
  set(CUDAQ_ENABLE_PYTHON ON CACHE BOOL "" FORCE)
endif()

# Backward compatibility: map the old CUDAQ_ENABLE_STATIC_LINKING to the two
# new flags. Check both -D (cmake variable) and environment variable paths.
# Setting cache variables here means the option() calls below will respect
# them (option() does not overwrite existing cache entries).
if(NOT DEFINED CUDAQ_ENABLE_STATIC_LINKING)
  set(CUDAQ_ENABLE_STATIC_LINKING "$ENV{CUDAQ_ENABLE_STATIC_LINKING}")
endif()
if(CUDAQ_ENABLE_STATIC_LINKING)
  if(NOT DEFINED CUDAQ_STATIC_CXX_RUNTIME)
    set(CUDAQ_STATIC_CXX_RUNTIME ON CACHE BOOL "Set via deprecated CUDAQ_ENABLE_STATIC_LINKING")
  endif()
  if(NOT DEFINED CUDAQ_STATIC_DEPS)
    set(CUDAQ_STATIC_DEPS ON CACHE BOOL "Set via deprecated CUDAQ_ENABLE_STATIC_LINKING")
  endif()
  message(DEPRECATION "CUDAQ_ENABLE_STATIC_LINKING is deprecated. "
    "Use CUDAQ_STATIC_CXX_RUNTIME and/or CUDAQ_STATIC_DEPS instead.")
endif()
option(CUDAQ_STATIC_CXX_RUNTIME "Statically link libstdc++ and libgcc into CUDA-Q libraries." OFF)
option(CUDAQ_STATIC_DEPS "Prefer static archives for external dependencies (zlib, OpenSSL, curl, BLAS)." OFF)
option (CUDAQ_FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." FALSE)
if (${CUDAQ_FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
       add_compile_options (-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
       add_compile_options (-fcolor-diagnostics)
    endif ()
endif ()

add_compile_options(-Wno-error=deprecated-declarations)
# Use plain -Wno- (not -Wno-error=) so GCC silently ignores it when unsupported.
# Guard with check_cxx_compiler_flag so Apple Clang (which errors on unknown
# -Wno- options) doesn't see it if the warning doesn't exist there.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wno-uninitialized-const-pointer
                        CUDAQ_HAS_WNO_UNINITIALIZED_CONST_POINTER)
if (CUDAQ_HAS_WNO_UNINITIALIZED_CONST_POINTER)
  add_compile_options(-Wno-uninitialized-const-pointer)
endif()
# GCC 12 headers emit false-positive diagnostics (e.g. char_traits.h,
# stl_algobase.h) when compiled with Clang using GCC's sysroot.
check_cxx_compiler_flag(-Wno-restrict CUDAQ_HAS_WNO_RESTRICT)
if (CUDAQ_HAS_WNO_RESTRICT)
  add_compile_options(-Wno-restrict)
endif()
# -Wstringop-overflow and -Wstringop-overread are GCC-only; check_cxx_compiler_flag caching is unreliable here.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-Wno-stringop-overflow -Wno-stringop-overread)
endif()

# Certain build configurations may be set directly in the environment.
# This facilitates some of the packaging (e.g. python packages built based on the pyproject.toml).
# These are cached so they persist across cmake runs without needing the env vars set again.
if(NOT LLVM_DIR AND EXISTS "$ENV{LLVM_INSTALL_PREFIX}/lib/cmake/llvm")
  SET(LLVM_DIR "$ENV{LLVM_INSTALL_PREFIX}/lib/cmake/llvm" CACHE PATH "Path to LLVM CMake config")
endif()
if(NOT ZLIB_ROOT)
  SET(ZLIB_ROOT "$ENV{ZLIB_INSTALL_PREFIX}" CACHE PATH "Path to zlib installation")
endif()
if(NOT OPENSSL_ROOT_DIR)
  SET(OPENSSL_ROOT_DIR "$ENV{OPENSSL_INSTALL_PREFIX}" CACHE PATH "Path to OpenSSL installation")
endif()
if(DEFINED ENV{OPENSSL_INSTALL_PREFIX} AND NOT "$ENV{OPENSSL_INSTALL_PREFIX}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH "$ENV{OPENSSL_INSTALL_PREFIX}")
endif()
if(NOT CURL_ROOT AND EXISTS "$ENV{CURL_INSTALL_PREFIX}/lib/cmake/CURL/CURLConfig.cmake")
  # Point to custom curl installation - CMake will use its config files
  SET(CURL_ROOT "$ENV{CURL_INSTALL_PREFIX}" CACHE PATH "Path to curl installation")
  SET(CMAKE_USE_SYSTEM_CURL TRUE)
endif()
if(NOT AWSSDK_ROOT AND CUDAQ_ENABLE_BRAKET_BACKEND)
  SET(AWSSDK_ROOT "$ENV{AWS_INSTALL_PREFIX}" CACHE PATH "Path to AWS SDK installation")
endif()
if(NOT QRMI_INSTALL_PREFIX AND CUDAQ_ENABLE_PASQAL_QRMI_CONNECTOR)
  SET(QRMI_INSTALL_PREFIX "$ENV{QRMI_INSTALL_PREFIX}" CACHE PATH "Path to QRMI installation")
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
if(APPLE)
  SET(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/lib;@loader_path/lib/plugins;@loader_path/../lib;@loader_path/../lib/plugins;@loader_path/../cudaq/mlir/_mlir_libs;@loader_path/../python/cudaq/mlir/_mlir_libs;@executable_path;@executable_path/lib;@executable_path/lib/plugins;@executable_path/../lib;@executable_path/../lib/plugins;@executable_path/../cudaq/mlir/_mlir_libs;@executable_path/../python/cudaq/mlir/_mlir_libs")
else()
  SET(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/lib/plugins:$ORIGIN/../lib:$ORIGIN/../lib/plugins:$ORIGIN/../cudaq/mlir/_mlir_libs:$ORIGIN/../python/cudaq/mlir/_mlir_libs")
endif()

# Platform-specific linker flags
# --no-as-needed forces linking of all specified libraries (GNU ld only, not available on Apple)
if(NOT APPLE)
  set(CUDAQ_FORCE_LINK_FLAG "-Wl,--no-as-needed")
else()
  set(CUDAQ_FORCE_LINK_FLAG "")
  # macOS Two-Level Namespace Workaround:
  # By default, macOS uses two-level namespace linking where symbols are bound
  # to the specific library they came from. This causes issues when LLVM/MLIR
  # static initializers (e.g., PassRegistry, TargetRegistry) are duplicated
  # across multiple shared libraries - each gets its own isolated copy.
  #
  # flat_namespace makes all symbols global, allowing shared libraries to see
  # each other's registrations. This is not a very graceful fix.
  # Alternatives might be:
  # A. Restructuring libraries to avoid multiple symbol definitions
  #    (but this still might struggle due to how plugin systems are used).
  # B. Using LLVM/MLIR dylibs. This won't work until later versions of LLVM are used as
  #    the first versions of this setting did not appropriately link all libraries to the dylibs.
  add_link_options("-Wl,-flat_namespace")
  add_link_options("-Wl,-undefined,dynamic_lookup")
endif()

# Detect sysroot for C++ stdlib headers/libs. Critical on macOS where a custom-built
# clang doesn't know the SDK location (unlike Linux where headers are in /usr/include).
if(NOT DEFINED CUDAQ_SYSROOT_PATH AND APPLE)
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE CUDAQ_SYSROOT_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()

if (CUDAQ_STATIC_CXX_RUNTIME)
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
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-libstdc++ -static-libgcc")
  message(STATUS "Static C++ runtime linking enabled.")
endif()

if (CUDAQ_STATIC_DEPS)
  set(BLA_STATIC ON)
  set(ZLIB_USE_STATIC_LIBS TRUE)
  set(OPENSSL_USE_STATIC_LIBS TRUE)
  set(CURL_USE_STATIC_LIBS TRUE)
  message(STATUS "Static external dependencies enabled.")
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
find_package(LLVM ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} CONFIG QUIET)

if(NOT LLVM_DIR)
  message(STATUS "LLVM_DIR not found for ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}, will try with llvm-config executable.")

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
    find_package(LLVM ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} REQUIRED CONFIG
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

# Detect C++ standard library headers for cudaq-quake
# cudaq-quake needs to find C++ stdlib headers for parsing user code.
# Priority order:
# 1. LLVM's bundled libc++ (if LLVM was built with LLVM_ENABLE_PROJECTS=libcxx)
# 2. macOS SDK sysroot (auto-detected via xcrun, provides system libc++)
# 3. System headers (Linux: /usr/include, works automatically)
# The paths are passed to cudaq-quake via nvqpp_config.h.in
set(CUDAQ_LIBCXX_PATH "${LLVM_BINARY_DIR}/include/c++/v1")
if(EXISTS "${CUDAQ_LIBCXX_PATH}")
  message(STATUS "Using LLVM libc++ headers: ${CUDAQ_LIBCXX_PATH}")
  set(CUDAQ_LIBCXX_TARGET_PATH "${LLVM_BINARY_DIR}/include/${LLVM_TARGET_TRIPLE}/c++/v1")
elseif(CUDAQ_SYSROOT_PATH)
  # Use sysroot (auto-detected on macOS, or user-provided)
  message(STATUS "Using sysroot for C++ headers: ${CUDAQ_SYSROOT_PATH}")
  unset(CUDAQ_LIBCXX_PATH)
else()
  # On Linux, system headers are found automatically
  unset(CUDAQ_LIBCXX_PATH)
endif()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

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
set(CUDAQ_MAIN_INCLUDE_DIR ${CUDAQ_MAIN_SRC_DIR}/cudaq/include)

set(CUDAQ_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CUDAQ_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CUDAQ_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CUDAQ_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cudaq/include)
set(CUDAQ_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
set(CUDAQ_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${CUDAQ_CMAKE_DIR}")

include(AddCUDAQ)
include(BuildHelpers)

# Enable sanitizers if requested (must be done early to affect all targets)
if(CUDAQ_ENABLE_SANITIZERS)
  include(Sanitizers)
  cudaq_enable_sanitizers()
endif()

# Add CUDA Quantum files to the include path
include_directories(BEFORE
  ${CUDAQ_SOURCE_DIR}/cudaq/include
  ${CUDAQ_BINARY_DIR}/cudaq/include)

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

configure_file("${CMAKE_SOURCE_DIR}/runtime/common/Version.cpp.in"
               "${CMAKE_BINARY_DIR}/runtime/common/Version.cpp" @ONLY)

# Check optional dependencies
# ==============================================================================

find_package(OpenSSL)
if (NOT CUDAQ_SKIP_MPI AND "runtime" IN_LIST CUDAQ_ENABLE_PROJECTS)
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
include_directories(SYSTEM tpls/json/single_include)
include_directories(SYSTEM tpls/json/single_include_fwd)
install(FILES tpls/json/single_include_fwd/nlohmann/json_fwd.hpp
  DESTINATION include/nlohmann)

# Add spdlog
set(SPDLOG_BUILD_SHARED OFF)
set(SPDLOG_FMT_EXTERNAL ON)
set(SPDLOG_BUILD_PIC ON)
set(SPDLOG_SYSTEM_INCLUDES ON)
add_subdirectory(tpls/spdlog)
target_compile_options(spdlog PRIVATE -Wno-covered-switch-default)

if (OPENSSL_FOUND AND CUDAQ_ENABLE_REST)
  # Look for CURL
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

# cuQuantum / cuTensor component discovery
if (CUDA_FOUND)
  find_package(cuStateVec)
  find_package(cuTensor)
  find_package(cuTensorNet)
  find_package(cuDensityMat)
endif()

set(CUDAQ_REALTIME_DIR "" CACHE PATH
    "Path to an installed cudaq-realtime prefix. Enables CUDA-Q realtime integration when set.")
set(CUDAQ_ENABLE_REALTIME FALSE)
if (CUDAQ_REALTIME_DIR)
  if (NOT CUDA_FOUND)
    message(FATAL_ERROR
      "CUDAQ_REALTIME_DIR requires CUDA support, but CUDA was not found.")
  endif()

  find_package(cudaq-realtime CONFIG REQUIRED
    PATHS "${CUDAQ_REALTIME_DIR}"
    NO_DEFAULT_PATH)

  if (NOT TARGET cudaq::cudaq-realtime)
    message(FATAL_ERROR
      "The realtime package at CUDAQ_REALTIME_DIR=${CUDAQ_REALTIME_DIR} "
      "does not define target cudaq::cudaq-realtime.")
  endif()
  if (NOT TARGET cudaq::cudaq-realtime-dispatch)
    message(FATAL_ERROR
      "The realtime package at CUDAQ_REALTIME_DIR=${CUDAQ_REALTIME_DIR} "
      "does not define target cudaq::cudaq-realtime-dispatch.")
  endif()

  set(CUDAQ_ENABLE_REALTIME TRUE)
  message(STATUS "CUDA-Q realtime integration enabled from: ${CUDAQ_REALTIME_DIR}")
else()
  message(STATUS
    "CUDA-Q realtime integration disabled. Set CUDAQ_REALTIME_DIR to an installed cudaq-realtime prefix to enable it.")
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
if("cudaq" IN_LIST CUDAQ_ENABLE_PROJECTS)
  add_subdirectory(cudaq)
endif()
if("runtime" IN_LIST CUDAQ_ENABLE_PROJECTS)
  add_subdirectory(runtime)
endif()
add_subdirectory(utils)

if(CUDAQ_BUILD_TESTS)
  enable_testing()
endif()

if("python" IN_LIST CUDAQ_ENABLE_PROJECTS)
  find_package(Python 3 COMPONENTS Interpreter Development)
  find_package(Python3 COMPONENTS Interpreter Development)

  add_subdirectory(tpls/nanobind)

  add_subdirectory(python)
endif()

# Add any external projects requested via CUDAQ_EXTERNAL_PROJECTS. Each entry's
# source directory must be supplied through the per-project cache variable
# `CUDAQ_EXTERNAL_<UPPER_NAME>_SOURCE_DIR`. The binary directory for each is
# placed under `${CMAKE_BINARY_DIR}/external/<name>`.
foreach(_cudaq_ext_proj IN LISTS CUDAQ_EXTERNAL_PROJECTS)
  string(TOUPPER "${_cudaq_ext_proj}" _cudaq_ext_upper)
  string(REGEX REPLACE "-" "_" _cudaq_ext_upper "${_cudaq_ext_upper}")
  set(_cudaq_ext_src "${CUDAQ_EXTERNAL_${_cudaq_ext_upper}_SOURCE_DIR}")
  if(NOT _cudaq_ext_src)
    message(FATAL_ERROR
      "CUDAQ_EXTERNAL_PROJECTS lists '${_cudaq_ext_proj}' but "
      "CUDAQ_EXTERNAL_${_cudaq_ext_upper}_SOURCE_DIR is not set.")
  endif()
  if(NOT EXISTS "${_cudaq_ext_src}/CMakeLists.txt")
    message(FATAL_ERROR
      "External project '${_cudaq_ext_proj}' source dir '${_cudaq_ext_src}' "
      "does not contain a CMakeLists.txt.")
  endif()
  message(STATUS "Adding external project '${_cudaq_ext_proj}' from ${_cudaq_ext_src}")
  add_subdirectory("${_cudaq_ext_src}"
                   "${CMAKE_BINARY_DIR}/external/${_cudaq_ext_proj}")
endforeach()

if(CUDAQ_BUILD_TESTS AND NOT CUDAQ_DISABLE_CPP_FRONTEND)
  include(CheckCXXCompilerFlag)
  umbrella_lit_testsuite_begin(check-all)
  set(INSTALL_GTEST OFF)
  add_subdirectory(tpls/googletest-src)
  # Turn off character-conversion warning in gtest for clang compilers
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-Wno-character-conversion CUDAQ_HAS_WNO_CHARACTER_CONVERSION)
  if (CUDAQ_HAS_WNO_CHARACTER_CONVERSION)
    target_compile_options(gtest PUBLIC -Wno-character-conversion)
  endif()
  # Bug in GCC 12 leads to spurious warnings (-Wrestrict)
  # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
  if (CMAKE_COMPILER_IS_GNUCXX
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0
    AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0.0)
    # Keep this GCC 12 workaround CXX-only. If it propagates to CUDA tests that
    # link gtest_main, such as test_gpu_get_state, nvcc may forward it to a
    # different CUDA host compiler where the option is invalid.
    target_compile_options(gtest PUBLIC $<$<COMPILE_LANGUAGE:CXX>:--param=evrp-mode=legacy>)
  endif()
  if (CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    target_compile_options(gtest PUBLIC -Wno-covered-switch-default)
    check_cxx_compiler_flag("-Wno-character-conversion"
                            CUDAQ_SUPPORTS_WNO_CHARACTER_CONVERSION)
    if (CUDAQ_SUPPORTS_WNO_CHARACTER_CONVERSION)
      target_compile_options(gtest PUBLIC -Wno-character-conversion)
    endif()
  endif()
  include(GoogleTest)
  include(CTest)
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
