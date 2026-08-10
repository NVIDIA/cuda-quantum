# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# CUDA-Q pulse builds either from the source tree as part of the top-level
# CUDA-Q build, or (mirroring cudaq-realtime) standalone as its own CMake
# project that installs a `cudaq-pulse` package consumable via
# find_package(cudaq-pulse CONFIG). Standalone builds must still be able to
# discover MLIR/LLVM (e.g. via MLIR_DIR) the same way the top-level build does.
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}"
       CUDAQ_PULSE_STANDALONE_BUILD)

if(CUDAQ_PULSE_STANDALONE_BUILD)
  cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

  # Set a default build type / install prefix if none was specified. Must be
  # set before project().
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel")
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.cudaq_pulse" CACHE STRING
      "Install path prefix, prepended onto install directories")

  project(cudaq-pulse LANGUAGES C CXX)

  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# CUDA-Q installers use a single relocatable library tree. Keep the pulse CMake
# package under that tree on distributions where GNUInstallDirs would otherwise
# default to lib64.
set(CMAKE_INSTALL_LIBDIR lib)

set(CUDAQ_PULSE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CUDAQ_PULSE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(CUDAQ_PULSE_FRONTEND_DIR
    "${CUDAQ_PULSE_SOURCE_DIR}/core/frontend/cudaq_pulse")
list(PREPEND CMAKE_MODULE_PATH "${CUDAQ_PULSE_SOURCE_DIR}/cmake")

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

# Installable CMake package (cudaq-pulse). Generated only when the GPU runtime
# was built, since cudaq::cudm_runtime is currently the exportable artifact;
# consuming the pulse dialects from an installed package is future work and can
# extend this export set. This lets core CUDA-Q (or any downstream) locate an
# installed pulse via find_package(cudaq-pulse CONFIG) / CUDAQ_PULSE_DIR.
if(TARGET cudm_runtime)
  set(CUDAQ_PULSE_HAS_CUDM TRUE)
  configure_package_config_file(
    ${CUDAQ_PULSE_SOURCE_DIR}/cmake/cudaq-pulse-config.cmake.in
    ${CUDAQ_PULSE_BINARY_DIR}/cmake/cudaq-pulse-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudaq-pulse)

  install(EXPORT cudaq-pulse-targets
    FILE cudaq-pulse-targets.cmake
    NAMESPACE cudaq::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudaq-pulse
    COMPONENT Development)

  install(FILES
    ${CUDAQ_PULSE_BINARY_DIR}/cmake/cudaq-pulse-config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudaq-pulse
    COMPONENT Development)
endif()
