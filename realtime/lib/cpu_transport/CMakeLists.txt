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
# Builds cudaq-realtime-cpu-roce-transport: the cpu_roce_* C ABI over
# Hololink's CpuRoceTransceiver. Used by:
#   - hsb_bridge_cpu (Phase 1, FPGA-facing bridge; tx_mode=kRdmaSend -- the
#     bridge Sends to the FPGA, which Writes its syndromes back)
#   - CpuRoceChannel (Phase 2, device_call caller / FPGA stand-in;
#     tx_mode=kRdmaWriteWithImm -- Writes requests like an FPGA)
#   - cpu_roce_test_daemon (Phase 2, service / bridge stand-in;
#     tx_mode=kRdmaSend -- Sends responses)
#
# It uses hololink::transport::roce, which links neither DOCA, Holoscan, CUDA,
# nor CUDA-dependent hololink::core, so the target remains buildable on hosts
# without GPUs.
# ==============================================================================

# ------------------------------------------------------------------------------
# UDP ring transceiver: carries the same ring contract as CpuRoceTransceiver
# over plain UDP (see udp_wrapper.h). No ibverbs, no CUDA -- always built.
# ------------------------------------------------------------------------------
add_library(cudaq-realtime-udp-transport STATIC
  udp_transceiver.cpp
)
add_library(cudaq::cudaq-realtime-udp-transport ALIAS cudaq-realtime-udp-transport)

target_include_directories(cudaq-realtime-udp-transport
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaq-realtime-udp-transport
  PUBLIC
    Threads::Threads
)

set_target_properties(cudaq-realtime-udp-transport PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

install(TARGETS cudaq-realtime-udp-transport
  EXPORT cudaq-realtime-targets
  COMPONENT realtime-lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

message(STATUS "  - cudaq-realtime-udp-transport (CPU UDP transport)")

# A full HSB build exports the CUDA-free leaf package in its build tree.
# HololinkRoce_DIR, HololinkRoce_ROOT, and CMAKE_PREFIX_PATH take
# precedence over this hint; default system prefixes do not.
find_package(HololinkRoce 2.7 CONFIG QUIET COMPONENTS transport
  HINTS "${HOLOSCAN_SENSOR_BRIDGE_BUILD_DIR}/HololinkRoce")
# CUDAQ_REALTIME_HAS_CPU_ROCE is read by cudaq-realtime-config.cmake.in,
# configured two directories up. INTERNAL forces the value, so each configure
# records the current answer.
if (NOT HololinkRoce_FOUND)
  set(CUDAQ_REALTIME_HAS_CPU_ROCE OFF CACHE INTERNAL "")
  message(STATUS
    "  - cpu_transport: HololinkRoce not found, skipping RoCE transport")
  return()
endif()
set(CUDAQ_REALTIME_HAS_CPU_ROCE ON CACHE INTERNAL "")
message(STATUS "Building cudaq-realtime-cpu-roce-transport with the following dependencies:")
message(STATUS "  HololinkRoce ${HololinkRoce_VERSION}: ${HololinkRoce_CONFIG}")

add_library(cudaq-realtime-cpu-roce-transport STATIC
  roce_wrapper_hololink.cpp
)
# Source-tree CUDA-Q integration uses the same cudaq:: target names
# that installed exports provide.
add_library(cudaq::cudaq-realtime-cpu-roce-transport ALIAS cudaq-realtime-cpu-roce-transport)

target_include_directories(cudaq-realtime-cpu-roce-transport
  PUBLIC
    $<BUILD_INTERFACE:${CUDAQ_REALTIME_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaq-realtime-cpu-roce-transport
  PUBLIC
    Threads::Threads
)
# PRIVATE: the C ABI header does not include Hololink. The static archive
# still exports LINK_ONLY hololink::transport::roce (and ibverbs).
target_link_libraries(cudaq-realtime-cpu-roce-transport
  PRIVATE
    hololink::transport::roce
)

set_target_properties(cudaq-realtime-cpu-roce-transport PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

install(TARGETS cudaq-realtime-cpu-roce-transport
  EXPORT cudaq-realtime-targets
  COMPONENT realtime-lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

message(STATUS
        "  - cudaq-realtime-cpu-roce-transport (CPU RoCE transport; Hololink)")
