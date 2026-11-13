# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Linear-ownership type interface and op traits, shared by every dialect with
# linear types.
add_cudaq_type_interface(LinearOwnershipInterfaces)

# Install only these two generated files: installing this whole binary
# directory would also pick up the CC, Quake and QEC subdirectories.
install(FILES
          ${CMAKE_CURRENT_BINARY_DIR}/LinearOwnershipInterfaces.h.inc
          ${CMAKE_CURRENT_BINARY_DIR}/LinearOwnershipInterfaces.cpp.inc
        DESTINATION include/cudaq/Optimizer/Dialect
        COMPONENT Development)

add_subdirectory(CC)
add_subdirectory(QEC)
add_subdirectory(Quake)
