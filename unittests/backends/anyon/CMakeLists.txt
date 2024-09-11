# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_anyon AnyonTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_anyon PRIVATE -Wl,--no-as-needed)
endif()
target_compile_definitions(test_anyon PRIVATE -DNVQIR_BACKEND_NAME=anyon)
target_include_directories(test_anyon PRIVATE ../..)
target_link_libraries(test_anyon
  PRIVATE fmt::fmt-header-only 
  cudaq-common 
  cudaq
  cudaq-builder
  cudaq-mlir-runtime
  cudaq-rest-qpu
  cudaq-spin 
  nvqir nvqir-qpp
  cudaq-platform-default 
  gtest_main)


configure_file("AnyonStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/anyon/AnyonStartServerAndTest.sh" @ONLY)
add_test(NAME anyon-tests COMMAND bash AnyonStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/anyon/)
