# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_dialect_library(OptimBuilder
  CUDAQBuilder.cpp
  Factory.cpp

  DEPENDS
    CCDialect
    QuakeDialect

  LINK_LIBS PUBLIC
    MLIRIR
    CCDialect
    QuakeDialect
)
