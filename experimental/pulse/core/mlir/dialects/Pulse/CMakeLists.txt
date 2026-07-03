# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(CudaqPulsePulse
  PulseDialect.cpp
  PulseOps.cpp
  PulseTypes.cpp

  DEPENDS
  CudaqPulsePulseIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRBytecodeOpInterface
)

