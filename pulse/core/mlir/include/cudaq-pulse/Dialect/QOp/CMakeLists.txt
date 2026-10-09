# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS QOpOps.td)
mlir_tablegen(QOpOps.h.inc -gen-op-decls)
mlir_tablegen(QOpOps.cpp.inc -gen-op-defs)
mlir_tablegen(QOpDialect.h.inc -gen-dialect-decls)
mlir_tablegen(QOpDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS QOpTypes.td)
mlir_tablegen(QOpTypes.h.inc -gen-typedef-decls)
mlir_tablegen(QOpTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS QOpAttrs.td)
mlir_tablegen(QOpAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(QOpAttrs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(QOpEnums.h.inc -gen-enum-decls)
mlir_tablegen(QOpEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(CudaqPulseQOpIncGen)
