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

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tpls/fmt
  OUTPUT_VARIABLE CUDAQ_FMTLIB_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Configure the file
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/NVQIRConfig.cmake.in
  ${CMAKE_BINARY_DIR}/cmake/Modules/NVQIRConfig.cmake
  @ONLY
)

install(FILES ${CONFIG_FILES} DESTINATION lib/cmake/cudaq)
install(FILES ${LANG_FILES} DESTINATION lib/cmake/cudaq)
install(FILES ${CMAKE_BINARY_DIR}/cmake/Modules/NVQIRConfig.cmake DESTINATION lib/cmake/nvqir)
# Note: we split the default target config generation into a separate file so that we can test it independently.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/NVQIRTargetConfigGen.cmake DESTINATION lib/cmake/nvqir)
