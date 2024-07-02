# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CONFIG_FILES
  CUDAQCommonConfig.cmake
  CUDAQEmDefaultConfig.cmake
  CUDAQNloptConfig.cmake
  CUDAQSpinConfig.cmake
  CUDAQConfig.cmake
  CUDAQEnsmallenConfig.cmake
  CUDAQPlatformDefaultConfig.cmake
)
set(LANG_FILES
  CMakeCUDAQCompiler.cmake.in
  CMakeCUDAQInformation.cmake
  CMakeDetermineCUDAQCompiler.cmake
  CMakeTestCUDAQCompiler.cmake
)

install(FILES ${CONFIG_FILES} DESTINATION lib/cmake/cudaq)
install(FILES ${LANG_FILES} DESTINATION lib/cmake/cudaq)
install(FILES NVQIRConfig.cmake DESTINATION lib/cmake/nvqir)
