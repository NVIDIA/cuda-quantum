# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_pasqal PasqalTester.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(test_pasqal PRIVATE -Wl,--no-as-needed)
endif()

target_include_directories(test_pasqal PRIVATE ../..)

target_link_libraries(test_pasqal
  PRIVATE
    fmt::fmt-header-only 
    cudaq
    cudaq-common 
    gtest_main
)

gtest_discover_tests(test_pasqal)
