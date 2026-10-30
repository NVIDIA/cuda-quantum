# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# GPU data plane through the bridge, linked at run time, on any GPU
# ==============================================================================
# gpu_loopback is a test-only provider whose rings are mapped pinned host
# memory, so test_gpu_dataplane drives a DEVICE unified dispatcher without a
# NIC.  It skips at run time without a GPU.

if (NOT CMAKE_CUDA_COMPILER OR NOT TARGET cudaq-realtime)
  return()
endif()

# The provider's hooks are its relocatable device code.  Like any provider it
# links no realtime library.
add_library(cudaq-realtime-bridge-gpu-loopback SHARED gpu_loopback_bridge.cu)
set_target_properties(cudaq-realtime-bridge-gpu-loopback PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_STANDARD 20)
target_include_directories(cudaq-realtime-bridge-gpu-loopback PRIVATE
  ${CUDAQ_REALTIME_INCLUDE_DIR})
target_link_libraries(cudaq-realtime-bridge-gpu-loopback PRIVATE CUDA::cudart)

# A binary without device code, for the reader's empty case.
add_library(gpu_dataplane_no_device_code SHARED no_device_code.cpp)
target_include_directories(gpu_dataplane_no_device_code PRIVATE
  ${CUDAQ_REALTIME_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS})

add_executable(test_gpu_dataplane
  test_gpu_dataplane.cpp
  test_gpu_dataplane_device.cu
  test_gpu_dataplane_second_handler.cu)
# The application's own dispatch kernel the test launches, through
# driver_launch.h.
cudaq_realtime_add_dispatch_kernel(test_gpu_dataplane
  ${CUDAQ_REALTIME_TEST_DISPATCHERS_DIR}/single_thread_dispatcher.cu)
target_include_directories(test_gpu_dataplane PRIVATE
  ${CUDAQ_REALTIME_TEST_DISPATCHERS_DIR})
set_target_properties(test_gpu_dataplane PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 20)
target_include_directories(test_gpu_dataplane PRIVATE
  ${CUDAQ_REALTIME_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(test_gpu_dataplane PRIVATE
  GPU_LOOPBACK_BRIDGE_PATH="$<TARGET_FILE:cudaq-realtime-bridge-gpu-loopback>"
  CUDAQ_REALTIME_TEST_EXPECT_CUDA_DEFINITION
  $<$<COMPILE_LANGUAGE:CUDA>:CUDAQ_REALTIME_TEST_CUDA_DEFINITION>)
target_link_libraries(test_gpu_dataplane PRIVATE
  GTest::gtest_main
  cudaq-realtime
  gpu_dataplane_no_device_code
  CUDA::cudart)

# The provider is dlopen'd, not linked.
add_dependencies(test_gpu_dataplane cudaq-realtime-bridge-gpu-loopback)

cudaq_gtest_discover_tests(test_gpu_dataplane DISCOVERY_TIMEOUT 120)
message(STATUS "  - test_gpu_dataplane (run-time linked GPU data plane)")
