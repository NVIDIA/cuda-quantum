# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_scaleway 
  SOURCES ScalewayTester.cpp
  BACKEND scaleway)

configure_file("ScalewayStartServerAndTest.sh.in" "${CMAKE_BINARY_DIR}/unittests/backends/scaleway/ScalewayStartServerAndTest.sh" @ONLY)
add_test(NAME scaleway-tests COMMAND bash ScalewayStartServerAndTest.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/backends/scaleway/)