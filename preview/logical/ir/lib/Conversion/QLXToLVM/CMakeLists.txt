# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_conversion_library(MLIRQLXToLVM
  QLXToLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/ir/include/qlx/Conversion

  DEPENDS
  MLIRQLXIncGen
  MLIRLVMIncGen
  MLIRQLXConversionPassesIncGen

  LINK_LIBS PUBLIC
  MLIRQLXDialect
  MLIRLVMDialect
  MLIRIR
  MLIRSupport
  MLIRArithDialect
)

qlx_propagate_llvm_rtti_eh(MLIRQLXToLVM)
