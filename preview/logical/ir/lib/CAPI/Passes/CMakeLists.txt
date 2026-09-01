# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# DISABLE_INSTALL keeps these out of MLIRTargets export set: they are
# embedded into the QLXPythonCAPI common library at build time.

add_mlir_public_c_api_library(QLXCAPIPasses
  Passes.cpp
  DISABLE_INSTALL

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  MLIRCAPIIR

  # Conversion
  MLIRQLXToLVM
  MLIRQuakeToQLX
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM

  # Dialect
  MLIRFabricTransforms
)
