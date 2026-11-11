# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# The linear-type interface, the linear-ownership op traits, and the analysis
# that enforces them. It names no dialect, so every dialect with linear types
# links it without a cycle.
qlx_select_mlir_link_libraries(_qlx_mlir_link_libraries MLIRIR MLIRSupport)

add_mlir_library(MLIRQLXLinearOwnership
  LinearOwnership.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/ir/include/qlx/Analysis

  DEPENDS
  MLIRQLXLinearOwnershipInterfacesIncGen

  LINK_LIBS PUBLIC
  ${_qlx_mlir_link_libraries}
)
qlx_propagate_llvm_rtti_eh(MLIRQLXLinearOwnership)
