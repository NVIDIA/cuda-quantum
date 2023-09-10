# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(CUDAQ_RELOCATABLE_BOOL "false")
if(CUDAQ_BUILD_RELOCATABLE_PACKAGE) 
  set(CUDAQ_RELOCATABLE_BOOL "true")
endif()

set(CUDAQ_IS_APPLE "false")
if (APPLE) 
  set(CUDAQ_IS_APPLE "true")
endif()

configure_file("nvq++.in"
               "${CUDAQ_BINARY_DIR}/bin/nvq++" @ONLY)

install(PROGRAMS ${CUDAQ_BINARY_DIR}/bin/nvq++
        DESTINATION bin)
install(FILES backendConfig.cpp DESTINATION targets/)
file(COPY backendConfig.cpp DESTINATION ${CMAKE_BINARY_DIR}/targets/)
