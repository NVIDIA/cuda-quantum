# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_library(MLIRFabricTransforms
  FabricCountPass.cpp
  FabricInlineRealisations.cpp
  FabricLowerProtocols.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/ir/include/qlx/Dialect/Fabric

  DEPENDS
  MLIRFabricIncGen
  MLIRFabricTransformPassesIncGen

  LINK_LIBS PUBLIC
  MLIRFabricDialect
  MLIRQLXDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)

qlx_propagate_llvm_rtti_eh(MLIRFabricTransforms)
