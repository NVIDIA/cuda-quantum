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
)

add_mlir_dialect_library(MLIRPhysDialect
  PhysDialect.cpp
  ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR}/ir/include/qlx/Dialect/Phys/IR
  DEPENDS MLIRPhysIncGen MLIREventIncGen
  LINK_LIBS PUBLIC MLIRFabricDialect MLIRQLXDialect MLIRCflowDialect MLIREventDialect ${_qlx_mlir_link_libraries})
