# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_record RecordParserTester.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_record PRIVATE -Wl,--no-as-needed)
endif()
target_include_directories(test_record PRIVATE ..)
target_link_libraries(test_record
  PRIVATE
    fmt::fmt-header-only 
    cudaq
    fmt::fmt-header-only 
    cudaq-common 
    gtest_main
    cudaq-platform-default)

gtest_discover_tests(test_record)
