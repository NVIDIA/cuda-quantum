# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_cudaq_library(OptAnalysis
  CircuitValidation.cpp
  UnitaryBuilder.cpp
  UnitaryOpGrouping.cpp

  DEPENDS
    CCDialect
    QuakeDialect
)

# UnitaryBuilder builds dense unitaries with Eigen (via runtime/common/EigenDense.h).
target_include_directories(OptAnalysis
  PRIVATE ${CMAKE_SOURCE_DIR}/tpls/eigen
  PRIVATE ${CMAKE_SOURCE_DIR}/runtime
)

register_cudaq_mlir_lib(OptAnalysis)
