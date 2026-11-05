# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

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
