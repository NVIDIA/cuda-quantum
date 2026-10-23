# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# GPU RoCE bridge and HSB playback tools
# ==============================================================================
# These targets are gated by CUDAQ_REALTIME_ENABLE_HSB_TOOLS 

# =========================================================================== #
# gpu_roce_bridge  (generic increment bridge)
# =========================================================================== #

message(STATUS "Building gpu_roce_bridge (generic increment)")

# Demo application that uses the gpu_roce bridge interface to implement a simple "increment" RPC function.
# This is the same as the one in cuda-quantum/realtime/unittests/utils 
# (which is implemented end-to-end without using the bridge pluggable interface).
add_executable(gpu_roce_app
  gpu_roce_bridge.cpp)

set_target_properties(gpu_roce_app PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(gpu_roce_app
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS})

# Note: no direct dependency on the HSB libraries needed.
target_link_libraries(gpu_roce_app
  PRIVATE
    cudaq-realtime-dispatch
    cudaq-realtime-bridge-gpu-roce
    cudaq-realtime)

# Find the rpc_increment_ft target built (device code for the increment function table) and link it to the app.
if (TARGET rpc_increment_ft)
  add_dependencies(gpu_roce_app rpc_increment_ft)
  message(STATUS "  - rpc_increment_ft target found, linking to app")
  target_link_libraries(gpu_roce_app PRIVATE rpc_increment_ft)
endif()

# Find the `hsb_fpga_emulator` and `hsb_fpga_playback` targets built by the utils/ CMakeLists.txt, and link them to the app for easy invocation.
if (TARGET hsb_fpga_emulator)
  add_dependencies(gpu_roce_app hsb_fpga_emulator)
  set(HSB_EMULATOR_BIN ${CMAKE_BINARY_DIR}/unittests/utils/hsb_fpga_emulator)
else()
  message(WARNING "hsb_fpga_emulator target not found. "
                  "FPGA emulation will not be available for gpu_roce_app.")
endif()
if (TARGET hsb_fpga_playback)
  add_dependencies(gpu_roce_app hsb_fpga_playback)
  set(HSB_PLAYBACK_BIN ${CMAKE_BINARY_DIR}/unittests/utils/hsb_fpga_playback)
else()
  message(WARNING "hsb_fpga_playback target not found. "
                  "FPGA playback will not be available for gpu_roce_app.")
endif()


configure_file(gpu_roce_test.sh.in gpu_roce_test.sh @ONLY)

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
