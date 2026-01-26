# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_oqc 
  SOURCES OQCTester.cpp
  BACKEND oqc
  )

configure_file("OQCStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/oqc/OQCStartServerAndTest.sh" @ONLY)
add_test(NAME oqc-tests COMMAND bash OQCStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/oqc/)
