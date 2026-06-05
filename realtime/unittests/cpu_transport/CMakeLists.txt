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
# Test artifacts that consume cudaq-realtime-cpu-transport:
#   - hsb_bridge_cpu              GPU-less bridge binary; mirrors hololink_bridge
#                                 but uses CpuRoceTransceiver +
#                                 CUDAQ_DISPATCH_HOST_CALL.
#   - hsb_test_cpu.sh             Orchestration script (built artifact; copy at
#                                 install time).
#
# All gated on TARGET cudaq-realtime-cpu-transport so this dir is silently
# skipped on hosts without libibverbs.
# ==============================================================================

if (NOT TARGET cudaq-realtime-cpu-transport)
  return()
endif()

# CUDA include path is needed because cudaq_realtime.h transitively includes
# <cuda_runtime.h> (for cudaGraphExec_t typedef in the function-entry union).
# We don't link CUDA runtime — only the headers.
find_package(CUDAToolkit REQUIRED)

# ----------------------------------------------------------------------------
# hsb_bridge_cpu: GPU-less bridge binary.
# ----------------------------------------------------------------------------
add_executable(hsb_bridge_cpu
  hsb_bridge_cpu.cpp
  init_rpc_increment_function_table_host.cpp
)

target_include_directories(hsb_bridge_cpu PRIVATE
  ${CUDAQ_REALTIME_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(hsb_bridge_cpu PRIVATE
  cudaq-realtime-cpu-transport     # CpuRoceTransceiver + C wrapper
  cudaq-realtime-host-dispatch     # cudaq_host_dispatcher_loop (HOST_CALL path)
  Threads::Threads
)

set_target_properties(hsb_bridge_cpu PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/utils
)

install(TARGETS hsb_bridge_cpu
  COMPONENT realtime-tests
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# ----------------------------------------------------------------------------
# hsb_test_cpu.sh: orchestration script, copied to the same output dir as the
# binary so the existing test-script convention (sibling of binary) works.
# ----------------------------------------------------------------------------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hsb_test_cpu.sh
               ${CMAKE_BINARY_DIR}/unittests/utils/hsb_test_cpu.sh COPYONLY)

message(STATUS "  - hsb_bridge_cpu (GPU-less Phase 1 bridge)")
