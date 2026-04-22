# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_qbraid
  SOURCES QbraidTester.cpp
  BACKEND qbraid
  BACKEND_CONFIG "qbraid emulate=false url=http://localhost:62454 api_key=00000000000000000000000000000000"
)

configure_file("QbraidStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/qbraid/QbraidStartServerAndTest.sh" @ONLY)
add_test(NAME qbraid-tests COMMAND bash QbraidStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/qbraid/)
