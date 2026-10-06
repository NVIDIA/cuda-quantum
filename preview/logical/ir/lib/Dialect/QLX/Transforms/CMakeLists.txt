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
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
)

add_mlir_library(MLIRQLXTransforms
  QLXAbsorbCliffordFrame.cpp
  QLXEstimateLogical.cpp
  QLXVerifyPBC.cpp
  QLXSynthesize.cpp
  QLXToPBC.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/ir/include/qlx/Dialect/QLX/Transforms

  DEPENDS
  MLIRQLXIncGen
  MLIRQLXTransformPassesIncGen

  LINK_LIBS PUBLIC
  MLIRQLXDialect
  MLIRCflowDialect
  cudaq-synth
  ${_qlx_mlir_link_libraries}
)

qlx_propagate_llvm_rtti_eh(MLIRQLXTransforms)
