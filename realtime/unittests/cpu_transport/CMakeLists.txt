# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

# ==============================================================================
# CPU RoCE transport test binaries + helpers (Phase 1)
#
# The shared validation receiver (roce_app) is built in realtime/validation.

# ----------------------------------------------------------------------------
# test_udp_transceiver: loopback tests for the UDP ring transceiver. Pure CPU
# (no CUDA, no ibverbs), so it builds and runs everywhere. Gated only on the
# always-built UDP target.
# ----------------------------------------------------------------------------
if (TARGET cudaq-realtime-udp-transport)
  add_executable(test_udp_transceiver test_udp_transceiver.cpp)

  target_include_directories(test_udp_transceiver PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
  )

  target_link_libraries(test_udp_transceiver PRIVATE
    GTest::gtest_main
    cudaq-realtime-udp-transport
    Threads::Threads
  )

  gtest_discover_tests(test_udp_transceiver
    TEST_PREFIX "test_udp_transceiver."
  )
  message(STATUS "  - test_udp_transceiver (UDP loopback ring transceiver)")
endif()

if (NOT TARGET cudaq-realtime-cpu-roce-transport)
  return()
endif()

add_executable(test_cpu_roce_hololink_wrapper
  test_cpu_roce_hololink_wrapper.cpp)
target_include_directories(test_cpu_roce_hololink_wrapper PRIVATE
  ${CUDAQ_REALTIME_INCLUDE_DIR})
target_link_libraries(test_cpu_roce_hololink_wrapper PRIVATE
  GTest::gtest_main
  cudaq-realtime-cpu-roce-transport)
cudaq_gtest_discover_tests(test_cpu_roce_hololink_wrapper
  TEST_PREFIX "test_cpu_roce_hololink_wrapper.")
message(STATUS
  "  - test_cpu_roce_hololink_wrapper (Hololink adapter contract)")

# The contract cases run anywhere. The service cases probe the RoCE transport
# and skip (or fail, under CUDAQ_REALTIME_REQUIRE_ROCE_TESTS) when it is
# unavailable.
add_executable(test_cpu_roce_transceiver test_cpu_roce_transceiver.cpp)
target_include_directories(test_cpu_roce_transceiver PRIVATE
  ${CUDAQ_REALTIME_INCLUDE_DIR})
target_compile_definitions(test_cpu_roce_transceiver PRIVATE
  CUDAQ_REALTIME_REQUIRE_ROCE_TESTS=$<BOOL:${CUDAQ_REALTIME_REQUIRE_ROCE_TESTS}>)
target_link_libraries(test_cpu_roce_transceiver PRIVATE
  GTest::gtest_main
  cudaq-realtime-cpu-roce-transport
  Threads::Threads)
cudaq_gtest_discover_tests(test_cpu_roce_transceiver
  TEST_PREFIX "test_cpu_roce_transceiver.")
message(STATUS
  "  - test_cpu_roce_transceiver (unified-mode hooks; service cases need RDMA)")
