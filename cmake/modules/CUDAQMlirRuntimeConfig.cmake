# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_filename_component(CUDAQ_MLIR_RUNTIME_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

if(NOT TARGET cudaq::cudaq-mlir-runtime)
  include("${CUDAQ_MLIR_RUNTIME_CMAKE_DIR}/CUDAQMlirRuntimeTargets.cmake")
endif()
