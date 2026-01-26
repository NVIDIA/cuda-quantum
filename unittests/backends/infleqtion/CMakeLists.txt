# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_infleqtion 
  SOURCES InfleqtionTester.cpp
  BACKEND infleqtion)

configure_file("InfleqtionStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/infleqtion/InfleqtionStartServerAndTest.sh" @ONLY)
add_test(NAME infleqtion-tests COMMAND bash InfleqtionStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/infleqtion/)
