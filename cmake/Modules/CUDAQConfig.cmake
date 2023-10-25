# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_filename_component(CUDAQ_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

include(CMakeFindDependencyMacro)
list(APPEND CMAKE_MODULE_PATH "${CUDAQ_CMAKE_DIR}")

set (CUDAQSpin_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQSpin REQUIRED)

set (CUDAQCommon_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQCommon REQUIRED)

set (CUDAQEmDefault_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQEmDefault REQUIRED)

set (CUDAQPlatformDefault_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQPlatformDefault REQUIRED)

set (CUDAQNlopt_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQNlopt REQUIRED)

set (CUDAQEnsmallen_DIR "${CUDAQ_CMAKE_DIR}")
find_dependency(CUDAQEnsmallen REQUIRED)

get_filename_component(PARENT_DIRECTORY ${CUDAQ_CMAKE_DIR} DIRECTORY)
get_filename_component(CUDAQ_LIBRARY_DIR ${PARENT_DIRECTORY} DIRECTORY)
get_filename_component(CUDAQ_INSTALL_DIR ${CUDAQ_LIBRARY_DIR} DIRECTORY)
set(CUDAQ_INCLUDE_DIR ${CUDAQ_INSTALL_DIR}/include)

set (NVQIR_DIR "${PARENT_DIRECTORY}/nvqir")
find_dependency(NVQIR REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)

if (NOT CUDAQ_LIBRARY_MODE)
  enable_language(CUDAQ)
endif() 

if(NOT TARGET cudaq::cudaq)
    include("${CUDAQ_CMAKE_DIR}/CUDAQTargets.cmake")
endif()

add_library(cudaq::cudaq-builder SHARED IMPORTED)
set_target_properties(cudaq::cudaq-builder PROPERTIES
  IMPORTED_LOCATION "${CUDAQ_LIBRARY_DIR}/libcudaq-builder${CMAKE_SHARED_LIBRARY_SUFFIX}"
  IMPORTED_SONAME "libcudaq-builder${CMAKE_SHARED_LIBRARY_SUFFIX}")