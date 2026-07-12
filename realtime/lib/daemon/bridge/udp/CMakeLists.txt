# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# UDP bridge provider: the udp ring transceiver behind the transport-provider
# interface.  Loaded at runtime via CUDAQ_REALTIME_BRIDGE_LIB /
# CUDAQ_PROVIDER_EXTERNAL; consumers speak only cudaq_bridge_* and stay
# transport-agnostic.  No ibverbs at link time; CUDA::cudart is linked only
# for the optional --pinned-rings mode (no GPU is touched unless that flag
# is passed).

add_library(cudaq-realtime-bridge-udp SHARED bridge_impl.cpp)

target_include_directories(cudaq-realtime-bridge-udp
  PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cudaq-realtime-bridge-udp
  PRIVATE
    cudaq-realtime-udp-transport
    Threads::Threads
    # --pinned-rings allocates CUDA pinned+mapped ring memory for GPU
    # consumers; no GPU is touched unless the flag is used.
    CUDA::cudart
)

set_target_properties(cudaq-realtime-bridge-udp PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

install(TARGETS cudaq-realtime-bridge-udp
  COMPONENT realtime-lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

message(STATUS "  - cudaq-realtime-bridge-udp (UDP bridge provider)")
