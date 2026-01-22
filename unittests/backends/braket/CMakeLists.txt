# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_braket 
  SOURCES BraketTester.cpp
  BACKEND braket)

configure_file("BraketStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/braket/BraketStartServerAndTest.sh" @ONLY)
add_test(NAME braket-tests COMMAND bash BraketStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/braket/)
