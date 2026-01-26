# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_qci 
  SOURCES QCITester.cpp
  BACKEND qci
  )

configure_file("QCIStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/qci/QCIStartServerAndTest.sh" @ONLY)
add_test(NAME qci-tests COMMAND bash QCIStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/qci/)
