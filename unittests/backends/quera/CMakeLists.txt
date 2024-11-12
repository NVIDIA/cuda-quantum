# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_quera JsonPayloadTester.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_quera PRIVATE -Wl,--no-as-needed)
endif()
target_include_directories(test_quera PRIVATE ../..)
target_link_libraries(test_quera PRIVATE 
    fmt::fmt-header-only 
    cudaq-common 
    gtest_main
)

gtest_discover_tests(test_quera)
