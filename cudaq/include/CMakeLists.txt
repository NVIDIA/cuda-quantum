# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/nvqpp_config.h.in"
               "${CUDAQ_BINARY_DIR}/cudaq/include/nvqpp_config.h"
               @ONLY)
add_subdirectory(cudaq)

install(DIRECTORY cudaq
        DESTINATION include
        FILES_MATCHING
          PATTERN "*.h"
          PATTERN "*.td")
