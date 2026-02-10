# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Hybrid test. Build the executable like any of the other tools, but
# run it like it was a lit test.
set(TEST_NAME test_argument_conversion)

add_llvm_executable(${TEST_NAME} test_argument_conversion.cpp)

target_compile_options(${TEST_NAME} PUBLIC -Wno-type-limits -fexceptions -DCUDAQ_RTTI_DISABLED)

target_include_directories(${TEST_NAME}
  PUBLIC
    ${CMAKE_SOURCE_DIR}/runtime
)

link_directories(${CMAKE_BINARY_DIR}/lib)

target_link_libraries(${TEST_NAME}
  PUBLIC
    cudaq-mlir-runtime
    cudaq
    cudaq-platform-default)

set_property(TARGET ${TEST_NAME} PROPERTY FOLDER test)
