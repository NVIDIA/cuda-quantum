# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_conversion_library(CudaqPulseCuDensityMatToLLVM
  CuDensityMatToLLVM.cpp

  DEPENDS
  CudaqPulseCuDensityMatIncGen

  LINK_LIBS PUBLIC
  CudaqPulseCuDensityMat
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRTransforms
)
