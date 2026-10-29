# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS QLXDialect.td)
mlir_tablegen(QLXDialect.h.inc -gen-dialect-decls)
mlir_tablegen(QLXDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRQLXDialectIncGen)

set(LLVM_TARGET_DEFINITIONS QLXTypes.td)
mlir_tablegen(QLXTypes.h.inc -gen-typedef-decls)
mlir_tablegen(QLXTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRQLXTypesIncGen)

set(LLVM_TARGET_DEFINITIONS QLXAttrs.td)
mlir_tablegen(QLXAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(QLXAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRQLXAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS QLXAttrs.td)
mlir_tablegen(QLXEnums.h.inc -gen-enum-decls)
mlir_tablegen(QLXEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRQLXEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS QLXOps.td)
mlir_tablegen(QLXOps.h.inc -gen-op-decls)
mlir_tablegen(QLXOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRQLXOpsIncGen)
