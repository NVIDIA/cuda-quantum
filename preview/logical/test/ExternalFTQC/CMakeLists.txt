# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

cmake_minimum_required(VERSION 3.30)
project(cudaq-logical-test-ftqc LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

find_package(CUDAQLogical REQUIRED CONFIG)

include_directories(
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CUDAQ_LOGICAL_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

set(LLVM_TARGET_DEFINITIONS include/test_ftqc/TestFTQCDialect.td)
mlir_tablegen(include/test_ftqc/TestFTQCDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/test_ftqc/TestFTQCDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(TestFTQCDialectIncGen)

set(LLVM_TARGET_DEFINITIONS include/test_ftqc/TestFTQCOps.td)
mlir_tablegen(include/test_ftqc/TestFTQCOps.h.inc -gen-op-decls)
mlir_tablegen(include/test_ftqc/TestFTQCOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TestFTQCOpsIncGen)

add_mlir_dialect_library(TestFTQCDialect
  lib/TestFTQCDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include

  DEPENDS
  TestFTQCDialectIncGen
  TestFTQCOpsIncGen

  LINK_LIBS PUBLIC
  CUDAQLogical::MLIRCUDAQLogicalInterfaces
)

add_library(test_ftqc_plugin SHARED lib/TestFTQCPlugin.cpp)
target_link_libraries(test_ftqc_plugin PRIVATE TestFTQCDialect)
target_compile_options(test_ftqc_plugin PRIVATE
  -fvisibility=hidden
  -fvisibility-inlines-hidden
)
llvm_update_compile_flags(test_ftqc_plugin)
set_target_properties(test_ftqc_plugin PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
