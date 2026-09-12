# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS LVMDialect.td)
mlir_tablegen(LVMDialect.h.inc -gen-dialect-decls)
mlir_tablegen(LVMDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRLVMDialectIncGen)

set(LLVM_TARGET_DEFINITIONS LVMTypes.td)
mlir_tablegen(LVMTypes.h.inc -gen-typedef-decls)
mlir_tablegen(LVMTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRLVMTypesIncGen)

set(LLVM_TARGET_DEFINITIONS LVMAttrs.td)
mlir_tablegen(LVMAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(LVMAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRLVMAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS LVMOps.td)
mlir_tablegen(LVMOps.h.inc -gen-op-decls)
mlir_tablegen(LVMOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRLVMOpsIncGen)
