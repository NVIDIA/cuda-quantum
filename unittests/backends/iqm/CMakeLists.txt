# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_iqm IQMTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_iqm PRIVATE -Wl,--no-as-needed)
endif()
target_compile_definitions(test_iqm PRIVATE -DNVQIR_BACKEND_NAME=iqm)
target_include_directories(test_iqm PRIVATE ../..)
target_link_libraries(test_iqm
  PRIVATE fmt::fmt-header-only
  cudaq-common
  cudaq
  cudaq-builder
  cudaq-mlir-runtime
  cudaq-rest-qpu
  cudaq-spin
  cudaq-platform-default
  gmock_main
  gtest_main)


configure_file("IQMStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/iqm/IQMStartServerAndTest.sh" @ONLY)
add_test(NAME iqm-tests COMMAND ./IQMStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/iqm/)
set_property(TEST iqm-tests PROPERTY SKIP_RETURN_CODE 77)
