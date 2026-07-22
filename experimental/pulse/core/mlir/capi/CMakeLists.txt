# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_public_c_api_library(CudaqPulseCAPI
  PulseCanonicalize.cpp

  LINK_LIBS PUBLIC
  CudaqPulsePulse
  MLIRTransforms
  MLIRCAPIIR
)
