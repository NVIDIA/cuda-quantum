# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_executable(test_target_config TargetConfigTester.cpp)

target_link_libraries(test_target_config
  PRIVATE
    CUDAQTargetConfigYaml
    gtest_main
)

cudaq_gtest_discover_tests(test_target_config DISCOVERY_TIMEOUT 120)
