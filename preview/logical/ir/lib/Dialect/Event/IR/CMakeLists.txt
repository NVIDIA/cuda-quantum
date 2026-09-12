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

add_mlir_dialect_library(MLIREventDialect
  EventDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/ir/include/qlx/Dialect/Event/IR

  DEPENDS
  MLIREventIncGen

  LINK_LIBS PUBLIC
  ${_qlx_mlir_link_libraries}
)
