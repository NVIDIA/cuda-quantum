# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Core-only installs support native backends without the CUDA-Q compiler or
# frontend execution platform. Only targets supplied by core are imported here.
include(CMakeFindDependencyMacro)
get_filename_component(CUDAQ_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(CUDAQ_INSTALL_DIR "${CUDAQ_CMAKE_DIR}/../../.." ABSOLUTE)
set(CUDAQ_LIBRARY_DIR "${CUDAQ_INSTALL_DIR}/lib")
set(CUDAQ_INCLUDE_DIR "${CUDAQ_INSTALL_DIR}/include")
find_dependency(CUDAQOperator HINTS "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQLogger HINTS "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQCommon HINTS "${CUDAQ_CMAKE_DIR}")
find_dependency(NVQIR HINTS "${CUDAQ_CMAKE_DIR}/../nvqir")
include("${CUDAQ_CMAKE_DIR}/CUDAQCoreTargets.cmake")
include("${CUDAQ_CMAKE_DIR}/CUDAQPythonBindingsConfig.cmake")
