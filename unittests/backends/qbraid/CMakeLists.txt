# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_qbraid QbraidTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_qbraid PRIVATE -Wl,--no-as-needed)
endif()
target_compile_definitions(test_qbraid PRIVATE -DNVQIR_BACKEND_NAME=qbraid)
target_include_directories(test_qbraid PRIVATE ../..)
target_link_libraries(test_qbraid
  PRIVATE fmt::fmt-header-only
  cudaq-common
  cudaq
  cudaq-builder
  cudaq-mlir-runtime
  cudaq-rest-qpu
  cudaq-platform-default
  gtest_main)


configure_file("QbraidStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/qbraid/QbraidStartServerAndTest.sh" @ONLY)
add_test(NAME qbraid-tests COMMAND bash QbraidStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/qbraid/)
