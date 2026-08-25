# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# UDP bridge provider tests
# ==============================================================================

# Functional: load the built provider by absolute path and bring a plain
# (non-pinned) udp bridge up through the v2 queries.  Links only the bridge
# loader (cudaq-realtime), NOT the CUDA runtime, so a regressed dynamic cudart
# dependency in the provider fails this test on a CUDA-runtime-less host.
add_executable(test_udp_bridge_provider test_udp_bridge_provider.cpp)

target_compile_definitions(test_udp_bridge_provider PRIVATE
  UDP_BRIDGE_PROVIDER_PATH="$<TARGET_FILE:cudaq-realtime-bridge-udp>"
)

target_link_libraries(test_udp_bridge_provider PRIVATE
  GTest::gtest_main
  cudaq-realtime
)

cudaq_gtest_discover_tests(test_udp_bridge_provider DISCOVERY_TIMEOUT 120)
