# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CUDAQ_IS_APPLE "false")
if (APPLE) 
  set(CUDAQ_IS_APPLE "true")
endif()

add_custom_target(nvq++)
configure_file("nvq++.in"
               "${CUDAQ_BINARY_DIR}/bin/nvq++" 
               FILE_PERMISSIONS
	              OWNER_READ OWNER_WRITE OWNER_EXECUTE
	              GROUP_READ GROUP_EXECUTE
	              WORLD_READ WORLD_EXECUTE
               @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/include/nvqpp_config.h
        DESTINATION include/nvqpp)
install(PROGRAMS ${CUDAQ_BINARY_DIR}/bin/nvq++
        DESTINATION bin)
install(FILES backendConfig.cpp DESTINATION targets/)
file(COPY backendConfig.cpp DESTINATION ${CMAKE_BINARY_DIR}/targets/)
