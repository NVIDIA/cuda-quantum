# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_translation_library(MLIRFabricTranslate
  EmitStim.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/ir/include/qlx/Target/Fabric

  DEPENDS
  MLIRFabricIncGen

  LINK_LIBS PUBLIC
  MLIRFabricDialect
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)

qlx_propagate_llvm_rtti_eh(MLIRFabricTranslate)
