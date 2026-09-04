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
  CUDAQPythonBindingsConfig.cmake
)
set(LANG_FILES
  CMakeCUDAQCompiler.cmake.in
  CMakeCUDAQInformation.cmake
  CMakeDetermineCUDAQCompiler.cmake
  CMakeTestCUDAQCompiler.cmake
)
set(FIND_MODULE_FILES
  FindGMP.cmake
  FindMPFR.cmake
)

# The installed NVQIRConfig.cmake fetches fmt at the commit CUDA-Q was built
# against, so that out-of-tree simulator plugins compile against the same
# headers.
if(GIT_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" ls-files --stage tpls/fmt
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE _cudaq_fmt_gitlink
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_cudaq_fmt_gitlink MATCHES "^160000 ([0-9a-f]+) ")
    set(CUDAQ_FMTLIB_HASH "${CMAKE_MATCH_1}")
  endif()
endif()

if(NOT CUDAQ_FMTLIB_HASH)
  message(FATAL_ERROR
    "Could not determine the fmtlib commit to record in NVQIRConfig.cmake from "
    "the git index.")
endif()
message(STATUS "fmtlib commit set to ${CUDAQ_FMTLIB_HASH}")

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
install(FILES ${FIND_MODULE_FILES}
        DESTINATION lib/cmake/cudaq
        COMPONENT Development)
install(FILES ${CMAKE_BINARY_DIR}/cmake/modules/NVQIRConfig.cmake
        DESTINATION lib/cmake/nvqir
        COMPONENT Development)
# Note: we split the default target config generation into a separate file so that we can test it independently.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/NVQIRTargetConfigGen.cmake
        DESTINATION lib/cmake/nvqir
        COMPONENT Development)

# Export CMake modules, metadata and helper scripts.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/AddCUDAQ.cmake
  DESTINATION lib/cmake/cudaq
  COMPONENT Development)
install(FILES ${CUDAQ_MLIR_BUNDLED_LIBS_PATH}
  DESTINATION lib/cmake/cudaq
  COMPONENT Development)
install(PROGRAMS ${CMAKE_SOURCE_DIR}/scripts/check_mlir_symbols.sh
  DESTINATION lib/cmake/cudaq
  COMPONENT Development)

# Optionally, bundle MLIR with the CUDA-Q installation.
include(${CMAKE_CURRENT_SOURCE_DIR}/BundleMLIRDevel.cmake)
