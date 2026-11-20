# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_cudaq_library(OptAnalysis
  CommutationAnalysis.cpp
  LinearOwnership.cpp
  QubitIdentityAnalysis.cpp
  UnitaryOpGrouping.cpp
  WireBorrowAnalysis.cpp

  DEPENDS
    CCDialect
    LinearOwnershipInterfacesIncGen
    QuakeDialect

  LINK_LIBS
    LinearOwnershipInterfaces
)

check_registered_mlir_lib(OptAnalysis)
