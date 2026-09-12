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
  MLIRBytecodeOpInterface
)

add_mlir_dialect_library(MLIRQLXDialect
  QLXDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/ir/include/qlx/Dialect/QLX/IR

  DEPENDS
  MLIRQLXIncGen
  MLIREventIncGen

  LINK_LIBS PUBLIC
  MLIREventDialect
  ${_qlx_mlir_link_libraries}
)

# Enable PIC so the dialect library can be linked into Python shared modules.
