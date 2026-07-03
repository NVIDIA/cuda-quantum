# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(CudaqPulseQOp
  QOpDialect.cpp
  QOpOps.cpp
  QOpTypes.cpp

  DEPENDS
  CudaqPulseQOpIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRBytecodeOpInterface
)
