# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Find MLIR installation
find_package(MLIR REQUIRED CONFIG)

# Find CUDA-Q installation  
set(CUDAQ_LIBRARY_MODE ON)
find_package(CUDAQ REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using CUDAQConfig.cmake in: ${CUDAQ_DIR}")

# Set up CMake module paths
set(CMAKE_MODULE_PATH
  ${LLVM_CMAKE_DIR}
  ${MLIR_CMAKE_DIR}
  ${CUDAQ_CMAKE_DIR}
  ${CMAKE_MODULE_PATH}
)

# Include necessary MLIR/LLVM modules
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Set up include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS}) 
include_directories(${CUDAQ_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add LLVM/MLIR definitions
add_definitions(${LLVM_DEFINITIONS})

# Get dialect and conversion libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

link_directories(${CUDAQ_INSTALL_DIR}/lib)

include(HandleLLVMOptions)
add_llvm_pass_plugin(DeviceCallShmem DeviceCallShmem.cpp)
target_link_libraries(DeviceCallShmem PRIVATE CCDialect)

set_target_properties(DeviceCallShmem PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
