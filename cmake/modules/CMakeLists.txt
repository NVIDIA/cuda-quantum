# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CONFIG_FILES
  CUDAQCommonConfig.cmake
  CUDAQEmDefaultConfig.cmake
  CUDAQNloptConfig.cmake
  CUDAQOperatorConfig.cmake
  CUDAQLoggerConfig.cmake
  CUDAQMlirRuntimeConfig.cmake
  CUDAQConfig.cmake
  CUDAQEnsmallenConfig.cmake
  CUDAQPlatformDefaultConfig.cmake
  CUDAQPythonInteropConfig.cmake
)
set(LANG_FILES
  CMakeCUDAQCompiler.cmake.in
  CMakeCUDAQInformation.cmake
  CMakeDetermineCUDAQCompiler.cmake
  CMakeTestCUDAQCompiler.cmake
)

find_package(Git QUIET)

# The installed NVQIRConfig.cmake fetches fmt at the commit CUDA-Q was built
# against, so that out-of-tree simulator plugins compile against the same
# headers. Resolving that commit must not assume that tpls/fmt is a usable git
# checkout: the wheel builds copy the sources into a container where the
# submodule's gitdir link no longer resolves, and `git rev-parse HEAD` then
# fails without any diagnostic, leaving GIT_TAG empty in the generated file.
set(CUDAQ_FMTLIB_HASH "" CACHE STRING
    "Commit of fmtlib that CUDA-Q is built against, baked into the installed NVQIRConfig.cmake. Detected from the tpls/fmt submodule when empty.")

if(NOT CUDAQ_FMTLIB_HASH)
  if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/tpls/fmt/.git")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/tpls/fmt"
      OUTPUT_VARIABLE CUDAQ_FMTLIB_HASH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
  endif()

  # Ask the CUDA-Q repository which commit the submodule is pinned to. This
  # works even when tpls/fmt carries no git metadata of its own.
  if(NOT CUDAQ_FMTLIB_HASH AND GIT_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD:tpls/fmt
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE CUDAQ_FMTLIB_HASH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
  endif()

  string(LENGTH "${CUDAQ_FMTLIB_HASH}" fmtlib_hash_length)
  if(NOT CUDAQ_FMTLIB_HASH MATCHES "^[0-9a-f]+$" OR NOT fmtlib_hash_length EQUAL 40)
    message(FATAL_ERROR
      "Could not determine the fmtlib commit to record in NVQIRConfig.cmake "
      "(got \"${CUDAQ_FMTLIB_HASH}\"). Initialize the tpls/fmt submodule, or "
      "configure with -DCUDAQ_FMTLIB_HASH=<commit>.")
  endif()
endif()
message(STATUS "NVQIRConfig.cmake will fetch fmtlib at ${CUDAQ_FMTLIB_HASH}")

# Configure the file
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/NVQIRConfig.cmake.in
  ${CMAKE_BINARY_DIR}/cmake/modules/NVQIRConfig.cmake
  @ONLY
)

install(FILES ${CONFIG_FILES}
        DESTINATION lib/cmake/cudaq
        COMPONENT Development)
install(FILES ${LANG_FILES}
        DESTINATION lib/cmake/cudaq
        COMPONENT Development)
install(FILES ${CMAKE_BINARY_DIR}/cmake/modules/NVQIRConfig.cmake
        DESTINATION lib/cmake/nvqir
        COMPONENT Development)
# Note: we split the default target config generation into a separate file so that we can test it independently.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/NVQIRTargetConfigGen.cmake
        DESTINATION lib/cmake/nvqir
        COMPONENT Development)

# Export CMake modules. AddCUDAQ.cmake resolves both the allowlist and the
# symbol-closure script relative to its own location, so all three have to land
# in the same directory.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/AddCUDAQ.cmake
        DESTINATION lib/cmake/cudaq
        COMPONENT Development)
install(FILES ${CMAKE_SOURCE_DIR}/cudaq/lib/Optimizer/mlir-libs-allowlist.txt
        DESTINATION lib/cmake/cudaq
        COMPONENT Development)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/scripts/check_mlir_symbols.sh
        DESTINATION lib/cmake/cudaq
        COMPONENT Development)

# Optionally, bundle MLIR with the CUDA-Q installation.
include(${CMAKE_CURRENT_SOURCE_DIR}/BundleMLIRDevel.cmake)
