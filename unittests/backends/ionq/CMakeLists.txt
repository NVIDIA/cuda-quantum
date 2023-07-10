# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_ionq IonQTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_ionq PRIVATE -Wl,--no-as-needed)
endif()
target_compile_definitions(test_ionq PRIVATE -DNVQIR_BACKEND_NAME=ionq)
target_include_directories(test_ionq PRIVATE ../..)
target_link_libraries(test_ionq
  PRIVATE fmt::fmt-header-only 
  cudaq-common 
  cudaq
  cudaq-builder
  cudaq-mlir-runtime
  cudaq-rest-qpu
  cudaq-spin 
  cudaq-platform-default 
  gtest_main)


configure_file("IonQStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/ionq/IonQStartServerAndTest.sh" @ONLY)
add_test(NAME ionq-tests COMMAND bash IonQStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/ionq/)
