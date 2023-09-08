# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_oqc OQCTester.cpp)
target_link_options(test_oqc PRIVATE -Wl,--no-as-needed)
target_compile_definitions(test_oqc PRIVATE -DNVQIR_BACKEND_NAME=oqc)
target_include_directories(test_oqc PRIVATE ../..)
target_link_libraries(test_oqc
  PRIVATE fmt::fmt-header-only 
  cudaq-common 
  cudaq
  cudaq-builder
  cudaq-mlir-runtime
  cudaq-rest-qpu
  cudaq-spin 
  cudaq-platform-default 
  gtest_main)


configure_file("OQCStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/oqc/OQCStartServerAndTest.sh" @ONLY)
add_test(NAME oqc-tests COMMAND bash OQCStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/oqc/)
