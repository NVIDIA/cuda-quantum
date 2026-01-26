# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_backend_unittest_executable(test_pasqal 
  SOURCES PasqalTester.cpp
  
  LINK_LIBS
  fmt::fmt-header-only 
  cudaq
  cudaq-common 
  gtest_main
  cudaq-platform-default
)

gtest_discover_tests(test_pasqal)
