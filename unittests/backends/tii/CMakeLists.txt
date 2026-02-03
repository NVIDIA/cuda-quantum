# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_tii 
  SOURCES TiiTester.cpp
  BACKEND tii
  )

configure_file("TiiStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/tii/TiiStartServerAndTest.sh" @ONLY)
add_test(NAME tii-tests COMMAND bash TiiStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/tii/)
