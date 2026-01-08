# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_qci QCITester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_qci PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
endif()
target_compile_definitions(test_qci PRIVATE -DNVQIR_BACKEND_NAME=qci)
target_include_directories(test_qci PRIVATE ../..)
target_link_libraries(test_qci
  PRIVATE fmt::fmt-header-only 
  cudaq-common 
  cudaq
  cudaq-builder
  cudaq-mlir-runtime
  cudaq-rest-qpu
  cudaq-operator
  cudaq-platform-default 
  gtest_main)

configure_file("QCIStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/qci/QCIStartServerAndTest.sh" @ONLY)
add_test(NAME qci-tests COMMAND bash QCIStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/qci/)
