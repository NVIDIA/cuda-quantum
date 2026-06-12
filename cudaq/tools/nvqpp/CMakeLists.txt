# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CUDAQ_IS_APPLE "false")
if (APPLE) 
  set(CUDAQ_IS_APPLE "true")
endif()

set(CUDAQ_DEVICE_CALL_RUNTIME_LINK_LIBS "")
if (CUDAQ_ENABLE_REALTIME)
  set(CUDAQ_DEVICE_CALL_RUNTIME_LINK_LIBS "-lcudaq-device-call-runtime")
endif()

add_custom_target(nvq++)
add_dependencies(nvq++ cudaq-builder nvqir-qpp)
configure_file("nvq++.in"
               "${CUDAQ_BINARY_DIR}/bin/nvq++" 
               FILE_PERMISSIONS
	              OWNER_READ OWNER_WRITE OWNER_EXECUTE
	              GROUP_READ GROUP_EXECUTE
	              WORLD_READ WORLD_EXECUTE
               @ONLY)

install(FILES ${CUDAQ_BINARY_DIR}/cudaq/include/nvqpp_config.h
        DESTINATION include/nvqpp)
install(PROGRAMS ${CUDAQ_BINARY_DIR}/bin/nvq++
        DESTINATION bin)
