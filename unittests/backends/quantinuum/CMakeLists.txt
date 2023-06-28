# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_quantinuum QuantinuumTester.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_quantinuum PRIVATE -Wl,--no-as-needed)
endif()
target_compile_definitions(test_quantinuum PRIVATE -DNVQIR_BACKEND_NAME=quantinuum)
target_include_directories(test_quantinuum PRIVATE ../..)
target_link_libraries(test_quantinuum
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


configure_file("QuantinuumStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/quantinuum/QuantinuumStartServerAndTest.sh" @ONLY)
add_test(NAME quantinuum-tests COMMAND ./QuantinuumStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/quantinuum/)
