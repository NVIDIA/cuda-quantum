# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# =========================================================================== #
# test_gpu_roce_two_process
# =========================================================================== #
# End-to-end: spawn the test server with a DEVICE dispatcher on a gpu_roce
# bridge and drive it from a cpu_roce caller in this process, under both
# dispatch shapes.  Needs RoCE and a GPU: every case probes the transport and
# skips (or fails, under CUDAQ_REALTIME_REQUIRE_ROCE_TESTS) when it is
# unavailable.
if (TARGET cudaq-realtime-test-server)
  get_target_property(_server_device_path cudaq-realtime-test-server
    CUDAQ_REALTIME_TEST_SERVER_DEVICE_PATH)
endif()
if (TARGET cudaq-realtime-bridge-gpu-roce AND
    _server_device_path AND
    TARGET cudaq-realtime-cpu-roce-transport)
  add_executable(test_gpu_roce_two_process test_gpu_roce_two_process.cpp)

  target_include_directories(test_gpu_roce_two_process PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_compile_definitions(test_gpu_roce_two_process PRIVATE
    CUDAQ_REALTIME_TEST_SERVER_PATH="$<TARGET_FILE:cudaq-realtime-test-server>"
    CUDAQ_REALTIME_REQUIRE_ROCE_TESTS=$<BOOL:${CUDAQ_REALTIME_REQUIRE_ROCE_TESTS}>
  )

  target_link_libraries(test_gpu_roce_two_process PRIVATE
    GTest::gtest_main
    cudaq-realtime-cpu-roce-transport
    CUDA::cudart
    Threads::Threads
  )

  # The server is exec'd and the provider dlopen'd by it, neither linked.
  add_dependencies(test_gpu_roce_two_process
    cudaq-realtime-test-server cudaq-realtime-bridge-gpu-roce)

  # Each case takes seconds. A hung case must not wait for the 1500 s default.
  cudaq_gtest_discover_tests(test_gpu_roce_two_process DISCOVERY_TIMEOUT 120
    PROPERTIES TIMEOUT 120)
  message(STATUS
    "  - test_gpu_roce_two_process (gpu_roce DEVICE dispatch, two processes)")
endif()
unset(_server_device_path)

configure_file(gpu_roce_test.sh.in gpu_roce_test.sh @ONLY)
