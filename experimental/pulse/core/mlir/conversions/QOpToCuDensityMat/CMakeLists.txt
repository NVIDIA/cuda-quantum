# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
