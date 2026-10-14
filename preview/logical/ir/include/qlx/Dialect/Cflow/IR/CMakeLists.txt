# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS CflowDialect.td)
mlir_tablegen(CflowDialect.h.inc -gen-dialect-decls)
mlir_tablegen(CflowDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRCflowDialectIncGen)

set(LLVM_TARGET_DEFINITIONS CflowOps.td)
mlir_tablegen(CflowOps.h.inc -gen-op-decls)
mlir_tablegen(CflowOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRCflowOpsIncGen)
