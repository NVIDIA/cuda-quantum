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

# The Clifford+T rotation-synthesis headers under cudaq/Synthesis back an
# internal-only C++ library (cudaq-synth). The public surface is the Python
# cudaq.synth module. cudaq-synth is not part of the exported CMake package,
# so excluding the headers keeps the install tree from implying a supported
# standalone C++ API.
install(DIRECTORY cudaq
        DESTINATION include
        COMPONENT Development
        FILES_MATCHING
          PATTERN "*.h"
          PATTERN "*.td"
          PATTERN "Synthesis" EXCLUDE)
