# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_conversion_library(CudaqPulseQOpToCuDensityMat
  QOpToCuDensityMat.cpp

  DEPENDS
  CudaqPulseQOpIncGen
  CudaqPulseCuDensityMatIncGen

  LINK_LIBS PUBLIC
  CudaqPulseQOp
  CudaqPulseCuDensityMat
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
)
