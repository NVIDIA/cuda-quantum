# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

qlx_select_mlir_link_libraries(_qlx_mlir_link_libraries
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)

add_mlir_translation_library(MLIRFabricTranslate
  EmitStim.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/ir/include/qlx/Target/Fabric

  DEPENDS
  MLIRFabricIncGen

  LINK_LIBS PUBLIC
  MLIRFabricDialect
  MLIRCflowDialect
  ${_qlx_mlir_link_libraries}
)

qlx_propagate_llvm_rtti_eh(MLIRFabricTranslate)
