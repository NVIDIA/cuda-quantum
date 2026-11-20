# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_cudaq_library(LinearOwnershipInterfaces
  LinearOwnershipInterfaces.cpp

  DEPENDS
  LinearOwnershipInterfacesIncGen

  LINK_LIBS
  MLIRIR
)

check_registered_mlir_lib(LinearOwnershipInterfaces)

add_subdirectory(CC)
add_subdirectory(QEC)
add_subdirectory(Quake)
