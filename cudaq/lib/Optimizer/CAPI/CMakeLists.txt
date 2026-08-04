# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_public_c_api_library(cudaqDialectsCAPI
  Dialects.cpp

  DEPENDS
  QuakeDialectIncGen
  QECOpsIncGen

  LINK_LIBS PRIVATE
  cudaqMLIR
)

add_cudaq_capi_shared_library(cudaqMLIRCAPI
  cudaqDialectsCAPI
  MLIRCAPIDebug
  MLIRCAPIExecutionEngine
  MLIRCAPIFunc
  MLIRCAPIIR
  MLIRCAPIInterfaces
  MLIRCAPILLVM
  MLIRCAPITarget
  MLIRCAPITransforms)

install(TARGETS cudaqMLIRCAPI
  EXPORT cudaq-targets
  COMPONENT CUDAQuantumPythonModules
  DESTINATION lib)
