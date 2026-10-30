# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_lqcloud
  SOURCES LQCloudTester.cpp
  BACKEND lqcloud
  BACKEND_CONFIG "lqcloud url=http://localhost:62453 qpu=mock_qpu")

configure_file(LQCloudStartServerAndTest.sh.in
               LQCloudStartServerAndTest.sh @ONLY)
add_test(NAME lqcloud-tests
  COMMAND bash LQCloudStartServerAndTest.sh $<TARGET_FILE:test_lqcloud>
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(lqcloud-tests PROPERTIES RESOURCE_LOCK lqcloud_port)
