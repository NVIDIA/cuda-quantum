# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(CudaqPulsePulseTransforms
  PulseVerify.cpp
  Canonicalize.cpp
  VirtualZ.cpp
  Fusion.cpp
  ScheduleAlap.cpp

  DEPENDS
  CudaqPulsePulseIncGen

  LINK_LIBS PUBLIC
  CudaqPulsePulse
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTransformUtils
)
