# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_infleqtion InfleqtionTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_infleqtion PRIVATE ${CUDAQ_FORCE_LINK_FLAG})
endif()
target_compile_definitions(test_infleqtion PRIVATE -DNVQIR_BACKEND_NAME=infleqtion)
target_include_directories(test_infleqtion PRIVATE ../..)
target_link_libraries(test_infleqtion
  PRIVATE fmt::fmt-header-only 
  cudaq-common 
  cudaq
  cudaq-builder
  cudaq-mlir-runtime
  cudaq-rest-qpu
  cudaq-operator
  cudaq-platform-default 
  gtest_main)


configure_file("InfleqtionStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/infleqtion/InfleqtionStartServerAndTest.sh" @ONLY)
add_test(NAME infleqtion-tests COMMAND bash InfleqtionStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/infleqtion/)
