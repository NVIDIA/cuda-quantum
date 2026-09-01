# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_library(MLIRQLXTransforms
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
  cudaq::MLIR
  MLIRIR
  MLIRSupport
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
)

qlx_propagate_llvm_rtti_eh(MLIRQLXTransforms)

# QLXSynthesize.cpp instantiates CUDA-Q's MPFR-backed synthesis templates.
target_link_libraries(MLIRQLXTransforms PUBLIC GMP::gmp MPFR::mpfr)
