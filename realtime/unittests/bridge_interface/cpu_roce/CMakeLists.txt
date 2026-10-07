# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# CPU RoCE bridge provider tests
# ==============================================================================

# End-to-end: spawn the transport-agnostic test server on a cpu_roce bridge
# and drive it from a caller transceiver in this process, under both dispatch
# shapes.  Needs RoCE: every case probes the transport and skips (or fails,
# under CUDAQ_REALTIME_REQUIRE_ROCE_TESTS) when it is unavailable.
if (TARGET cudaq-realtime-bridge-cpu-roce AND
    TARGET cudaq-realtime-test-server AND
    TARGET cudaq-realtime-cpu-roce-transport)
  add_executable(test_cpu_roce_two_process test_cpu_roce_two_process.cpp)

  target_include_directories(test_cpu_roce_two_process PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
  )

  target_compile_definitions(test_cpu_roce_two_process PRIVATE
    CUDAQ_REALTIME_TEST_SERVER_PATH="$<TARGET_FILE:cudaq-realtime-test-server>"
    CUDAQ_REALTIME_REQUIRE_ROCE_TESTS=$<BOOL:${CUDAQ_REALTIME_REQUIRE_ROCE_TESTS}>
  )

  target_link_libraries(test_cpu_roce_two_process PRIVATE
    GTest::gtest_main
    cudaq-realtime-cpu-roce-transport
    Threads::Threads
  )

  # The server is exec'd, not linked.
  add_dependencies(test_cpu_roce_two_process cudaq-realtime-test-server)

  cudaq_gtest_discover_tests(test_cpu_roce_two_process DISCOVERY_TIMEOUT 120)
  message(STATUS
    "  - test_cpu_roce_two_process (cpu_roce dispatch, two processes)")
endif()
