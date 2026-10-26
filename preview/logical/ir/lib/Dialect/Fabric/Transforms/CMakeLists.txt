# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

qlx_select_mlir_link_libraries(_qlx_mlir_link_libraries
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)

add_mlir_library(MLIRFabricTransforms
  FabricCountPass.cpp
  FabricEstimateAnalytical.cpp
  FabricInlineRealisations.cpp
  FabricLowerProtocols.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/ir/include/qlx/Dialect/Fabric

  DEPENDS
  MLIRFabricIncGen
  MLIRFabricTransformPassesIncGen
  MLIREventIncGen

  LINK_LIBS PUBLIC
  MLIRFabricDialect
  MLIRCflowDialect
  MLIRPhysDialect
  MLIRQLXDialect
  MLIREventDialect
  QLXSpacetimeProviders
  ${_qlx_mlir_link_libraries}
)

qlx_propagate_llvm_rtti_eh(MLIRFabricTransforms)
