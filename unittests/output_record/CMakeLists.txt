# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_record RecordParserTester.cpp)

target_include_directories(test_record PRIVATE ..)
target_link_libraries(test_record
  PRIVATE
    cudaq
    cudaq-common
    cudaq-logger
    gtest_main
    cudaq-platform-default)

cudaq_gtest_discover_tests(test_record DISCOVERY_TIMEOUT 120)
