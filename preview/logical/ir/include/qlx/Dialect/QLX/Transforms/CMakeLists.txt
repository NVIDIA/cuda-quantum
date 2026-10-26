# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name QLXTransform)
add_public_tablegen_target(MLIRQLXTransformPassesIncGen)

mlir_tablegen(QLXTransformPasses.md -gen-pass-doc)
add_custom_target(MLIRQLXTransformPassesDocGen
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/QLXTransformPasses.md)
set_target_properties(MLIRQLXTransformPassesDocGen PROPERTIES
  FOLDER "QLX/Tablegenning/Docs")
