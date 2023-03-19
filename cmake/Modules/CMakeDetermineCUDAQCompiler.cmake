# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

get_filename_component(CUDAQ_THIS_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(LOCAL_CUDAQ_CMAKE_DIR ${CUDAQ_THIS_CMAKE_DIR} DIRECTORY)
get_filename_component(CUDAQ_LIB_DIR ${LOCAL_CUDAQ_CMAKE_DIR} DIRECTORY)
get_filename_component(CUDAQ_INSTALL_DIR ${CUDAQ_LIB_DIR} DIRECTORY)

# Find the compiler
find_program(
    CMAKE_CUDAQ_COMPILER 
        NAMES "nvq++" 
        HINTS "${CUDAQ_INSTALL_DIR}/bin"
        DOC "The NVIDIA CUDA Quantum compiler - nvq++" 
)

message(STATUS "The CUDA Quantum compiler identification is NVQ++ - ${CMAKE_CUDAQ_COMPILER}")
mark_as_advanced(CMAKE_CUDAQ_COMPILER)

set(CMAKE_CUDAQ_SOURCE_FILE_EXTENSIONS cpp;cxx)
set(CMAKE_CUDAQ_OUTPUT_EXTENSION .o)
set(CMAKE_CUDAQ_COMPILER_ENV_VAR "QXX")

# Configure variables set in this file for fast reload later on
configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeCUDAQCompiler.cmake.in
               ${CMAKE_PLATFORM_INFO_DIR}/CMakeCUDAQCompiler.cmake)
