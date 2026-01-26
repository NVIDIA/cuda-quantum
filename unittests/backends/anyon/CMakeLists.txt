# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_anyon 
  SOURCES AnyonTester.cpp 
  BACKEND anyon
  )

configure_file("AnyonStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/anyon/AnyonStartServerAndTest.sh" @ONLY)
add_test(NAME anyon-tests COMMAND bash AnyonStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/anyon/)
