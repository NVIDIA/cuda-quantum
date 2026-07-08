# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_cudaq_dialect(Quake quake)
add_cudaq_interface(QuakeInterfaces)
add_cudaq_dialect_doc(QuakeDialect quake)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
        DESTINATION include/cudaq/Optimizer/Dialect/Quake
        FILES_MATCHING
          PATTERN "*.h.inc"
          PATTERN "*.cpp.inc"
          PATTERN "CMakeFiles" EXCLUDE)
