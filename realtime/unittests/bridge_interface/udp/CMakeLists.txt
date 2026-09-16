# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# UDP bridge provider tests
# ==============================================================================

# Everything here loads or drives the udp provider .so.
if (NOT TARGET cudaq-realtime-bridge-udp)
  return()
endif()

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

# End-to-end: spawn the transport-agnostic test server on a udp bridge and
# drive it from a caller transceiver in this process.  A separate binary from
# test_udp_bridge_provider on purpose -- that one links no transport and no CUDA
# runtime, which is what makes it catch a regressed cudart dependency in the
# provider.  This one needs the transceiver to speak the wire.
if (TARGET cudaq-realtime-test-server AND TARGET cudaq-realtime-udp-transport)
  add_executable(test_udp_two_process test_udp_two_process.cpp)

  target_include_directories(test_udp_two_process PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
  )

  target_compile_definitions(test_udp_two_process PRIVATE
    CUDAQ_REALTIME_TEST_SERVER_PATH="$<TARGET_FILE:cudaq-realtime-test-server>"
  )

  target_link_libraries(test_udp_two_process PRIVATE
    GTest::gtest_main
    cudaq-realtime-udp-transport
    Threads::Threads
  )

  # The server is exec'd, not linked.
  add_dependencies(test_udp_two_process cudaq-realtime-test-server)

  cudaq_gtest_discover_tests(test_udp_two_process DISCOVERY_TIMEOUT 120)
  message(STATUS "  - test_udp_two_process (UDP dispatch, two processes)")
endif()
