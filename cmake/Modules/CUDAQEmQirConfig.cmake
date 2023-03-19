# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_filename_component(CUDAQ_EM_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

set (CUDAQSpin_DIR "${CUDAQ_EM_CMAKE_DIR}")
find_dependency(CUDAQSpin REQUIRED)

set (CUDAQCommon_DIR "${CUDAQ_EM_CMAKE_DIR}")
find_dependency(CUDAQCommon REQUIRED)

if(NOT TARGET cudaq::cudaq-em-qir)
  include("${CUDAQ_EM_CMAKE_DIR}/CUDAQEmQirTargets.cmake")
endif()
