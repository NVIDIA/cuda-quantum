# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

# ==============================================================================
# CPU RoCE transport library (Phase 1 of cpu_transport umbrella)
#
# Builds cudaq-realtime-cpu-transport, a static library exporting the
# CpuRoceTransceiver class plus its extern-C wrapper. Used by:
#   - hsb_bridge_cpu (Phase 1, FPGA-facing bridge; tx_mode=kRdmaSend -- the
#     bridge Sends to the FPGA, which Writes its syndromes back)
#   - CpuRoceChannel (Phase 2, device_call caller / FPGA stand-in;
#     tx_mode=kRdmaWriteWithImm -- Writes requests like an FPGA)
#   - cpu_roce_test_daemon (Phase 2, service / bridge stand-in;
#     tx_mode=kRdmaSend -- Sends responses)
#
# Depends only on libibverbs + standard C++; deliberately no DOCA, no Hololink,
# no Holoscan, no CUDA at link time so the library is buildable on hosts
# without GPUs.
# ==============================================================================

find_library(IBVERBS_LIBRARY NAMES ibverbs)
if (NOT IBVERBS_LIBRARY)
  message(STATUS "  - cpu_transport: libibverbs not found, skipping")
  return()
endif()

add_library(cudaq-realtime-cpu-transport STATIC
  roce_transceiver.cpp  # class CpuRoceTransceiver: ibv setup, RX/TX threads, ring lifecycle
  roce_wrapper.cpp      # extern-C shim used by bridge tools and CpuRoceChannel
)
# Source-tree CUDA-Q integration uses the same cudaq:: target names
# that installed exports provide.
add_library(cudaq::cudaq-realtime-cpu-transport ALIAS cudaq-realtime-cpu-transport)

target_include_directories(cudaq-realtime-cpu-transport
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaq-realtime-cpu-transport
  PUBLIC
    ${IBVERBS_LIBRARY}
    Threads::Threads
)

set_target_properties(cudaq-realtime-cpu-transport PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

install(TARGETS cudaq-realtime-cpu-transport
  EXPORT cudaq-realtime-targets
  COMPONENT realtime-lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

message(STATUS "  - cudaq-realtime-cpu-transport (CPU RoCE transport; libibverbs)")
