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
# Builds cudaq-realtime-cpu-roce-transport. Both backends export the
# cpu_roce_* C ABI; only the legacy backend also provides CUDA-Q's
# CpuRoceTransceiver C++ implementation. Used by:
#   - hsb_bridge_cpu (Phase 1, FPGA-facing bridge; tx_mode=kRdmaSend -- the
#     bridge Sends to the FPGA, which Writes its syndromes back)
#   - CpuRoceChannel (Phase 2, device_call caller / FPGA stand-in;
#     tx_mode=kRdmaWriteWithImm -- Writes requests like an FPGA)
#   - cpu_roce_test_daemon (Phase 2, service / bridge stand-in;
#     tx_mode=kRdmaSend -- Sends responses)
#
# The legacy backend depends directly on libibverbs; the Hololink backend uses
# hololink::transport::roce. Neither uses DOCA, HSB libraries, Holoscan, or CUDA,
# so the target remains buildable on hosts without GPUs.
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

set(CUDAQ_REALTIME_CPU_ROCE_USES_HOLOLINK OFF CACHE INTERNAL
    "Whether the configured CPU RoCE backend uses Hololink" FORCE)

set(CUDAQ_REALTIME_CPU_ROCE_BACKEND "legacy" CACHE STRING
    "CPU RoCE backend: legacy or hololink (explicit opt-in)")
set_property(CACHE CUDAQ_REALTIME_CPU_ROCE_BACKEND PROPERTY STRINGS legacy hololink)

if(CUDAQ_REALTIME_CPU_ROCE_BACKEND STREQUAL "hololink")
  find_package(HololinkRoce 2.7 CONFIG REQUIRED COMPONENTS transport)
  set(_cudaq_cpu_roce_backend "hololink")
elseif(CUDAQ_REALTIME_CPU_ROCE_BACKEND STREQUAL "legacy")
  set(_cudaq_cpu_roce_backend "legacy")
else()
  message(FATAL_ERROR
          "CUDAQ_REALTIME_CPU_ROCE_BACKEND must be 'legacy' or 'hololink', "
          "got '${CUDAQ_REALTIME_CPU_ROCE_BACKEND}'")
endif()

if(_cudaq_cpu_roce_backend STREQUAL "hololink")
  set(_cudaq_cpu_roce_sources
      roce_wrapper_hololink.cpp)
  set(CUDAQ_REALTIME_CPU_ROCE_USES_HOLOLINK ON CACHE INTERNAL
      "Whether the configured CPU RoCE backend uses Hololink" FORCE)
else()
  find_library(IBVERBS_LIBRARY NAMES ibverbs)
  if(NOT IBVERBS_LIBRARY)
    message(STATUS
      "  - cpu_transport: libibverbs not found, skipping legacy RoCE transport")
    unset(_cudaq_cpu_roce_backend)
    return()
  endif()
  set(_cudaq_cpu_roce_sources
      roce_transceiver.cpp  # legacy ibv setup, RX/TX threads, ring lifecycle
      roce_wrapper.cpp)     # legacy extern-C shim
  set(CUDAQ_REALTIME_CPU_ROCE_USES_HOLOLINK OFF CACHE INTERNAL
      "Whether the configured CPU RoCE backend uses Hololink" FORCE)
endif()

add_library(cudaq-realtime-cpu-roce-transport STATIC
  ${_cudaq_cpu_roce_sources}
)
unset(_cudaq_cpu_roce_sources)
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
if(_cudaq_cpu_roce_backend STREQUAL "hololink")
  target_link_libraries(cudaq-realtime-cpu-roce-transport
    PUBLIC
      hololink::transport::roce
  )
else()
  target_link_libraries(cudaq-realtime-cpu-roce-transport
    PUBLIC
      ${IBVERBS_LIBRARY}
  )
endif()

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
        "  - cudaq-realtime-cpu-roce-transport "
        "(CPU RoCE transport; ${_cudaq_cpu_roce_backend} backend)")
unset(_cudaq_cpu_roce_backend)
