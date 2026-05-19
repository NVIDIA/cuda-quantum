# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_dialect_library(OptimBuilder
  Factory.cpp
  Intrinsics.cpp
  Marshal.cpp

  DEPENDS
    CCDialect
    QuakeDialect

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRParser
    CCDialect
    QuakeDialect
)
