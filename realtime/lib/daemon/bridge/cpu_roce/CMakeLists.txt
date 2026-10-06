# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# CPU RoCE bridge provider: the CPU RoCE RDMA ring transceiver behind the
# transport-provider interface, including both QP-exchange methods (TCP
# rendezvous with a CpuRoceChannel caller, and the HSB FPGA handshake).
# Loaded at runtime via CUDAQ_REALTIME_BRIDGE_LIB / CUDAQ_PROVIDER_EXTERNAL.
# Built only when the CPU RoCE transport itself was built (libibverbs found).

if (NOT TARGET cudaq-realtime-cpu-roce-transport)
  message(STATUS "  - cudaq-realtime-bridge-cpu-roce skipped (no CPU RoCE transport)")
  return()
endif()

add_library(cudaq-realtime-bridge-cpu-roce SHARED bridge_impl.cpp)

target_include_directories(cudaq-realtime-bridge-cpu-roce
  PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cudaq-realtime-bridge-cpu-roce
  PRIVATE
    cudaq-realtime-cpu-roce-transport
    Threads::Threads
)

set_target_properties(cudaq-realtime-bridge-cpu-roce PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

install(TARGETS cudaq-realtime-bridge-cpu-roce
  COMPONENT realtime-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

message(STATUS "  - cudaq-realtime-bridge-cpu-roce (CPU RoCE bridge provider)")
