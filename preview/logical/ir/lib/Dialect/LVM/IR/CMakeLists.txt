# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_mlir_dialect_library(MLIRLVMDialect
  LVMDialect.cpp
  ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR}/ir/include/qlx/Dialect/LVM/IR
  DEPENDS MLIRLVMIncGen
  LINK_LIBS PUBLIC MLIRQLXDialect MLIRIR MLIRSupport)
