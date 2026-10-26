# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(GNUInstallDirs)

install(DIRECTORY ${CUDAQ_REALTIME_INCLUDE_DIR}/cudaq
  COMPONENT cudaq-realtime-headers
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh"
)

# cpu_transport first: the daemon's bridge providers (bridge/udp,
# bridge/cpu_roce) link the transport targets defined there and probe them
# with if(TARGET ...).
add_subdirectory(cpu_transport)
add_subdirectory(daemon)
