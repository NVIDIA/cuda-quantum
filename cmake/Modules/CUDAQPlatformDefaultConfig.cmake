# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_filename_component(CUDAQ_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

set (CUDAQEmDefault_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQEmDefault REQUIRED)

set (CUDAQSpin_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQSpin REQUIRED)

set (CUDAQCommon_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQCommon REQUIRED)

if(NOT TARGET cudaq::cudaq-platform-default)
  include("${CUDAQ_CMAKE_DIR}/CUDAQPlatformDefaultTargets.cmake")
endif()
