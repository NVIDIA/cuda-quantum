# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# A simple example demonstrating how to use the CUDA Quantum Realtime library.
project(dispatch_kernel)
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

if(NOT CUDAQ_REALTIME_ROOT)
  SET(CUDAQ_REALTIME_ROOT "$ENV{CUDAQ_REALTIME_ROOT}" CACHE PATH "Path to CUDA Quantum Realtime installation")
endif()

if (NOT CUDAQ_REALTIME_ROOT)
  # Default path to look for the library if the environment variable is not set
    set(CUDAQ_REALTIME_ROOT "/opt/nvidia/cudaq/realtime" CACHE PATH "Default path to CUDA Quantum Realtime installation")
   message(STATUS "CUDAQ_REALTIME_ROOT environment variable is not set. Use the default path: ${CUDAQ_REALTIME_ROOT}")
endif()


find_library(CUDAQ_REALTIME_LIB 
    NAMES cudaq-realtime 
    HINTS   
        ${CUDAQ_REALTIME_ROOT}/lib
)

if (NOT CUDAQ_REALTIME_LIB)
  message(FATAL_ERROR "Could not find cudaq-realtime library. Please set CUDAQ_REALTIME_ROOT to the correct path.")
else()
  message(STATUS "Found cudaq-realtime library at: ${CUDAQ_REALTIME_LIB}")
endif()

find_library(CUDAQ_REALTIME_DISPATCH_LIB
    NAMES cudaq-realtime-dispatch 
    HINTS   
        ${CUDAQ_REALTIME_ROOT}/lib
)   

if (NOT CUDAQ_REALTIME_DISPATCH_LIB)
  message(FATAL_ERROR "Could not find cudaq-realtime-dispatch library. Please set CUDAQ_REALTIME_ROOT to the correct path.")
else()
  message(STATUS "Found cudaq-realtime-dispatch library at: ${CUDAQ_REALTIME_DISPATCH_LIB}")
endif()


add_executable(dispatch_kernel dispatch_kernel.cu)

set_target_properties(dispatch_kernel PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 17
)
set_target_properties(dispatch_kernel PROPERTIES CUDA_ARCHITECTURES "80;90")

target_include_directories(dispatch_kernel PRIVATE
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CUDAQ_REALTIME_ROOT}/include
)

target_link_libraries(dispatch_kernel PRIVATE 
    CUDA::cudart
    ${CUDAQ_REALTIME_LIB}
    ${CUDAQ_REALTIME_DISPATCH_LIB}
)
