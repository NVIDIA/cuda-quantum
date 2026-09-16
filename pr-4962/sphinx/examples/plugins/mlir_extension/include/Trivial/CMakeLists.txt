# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS TrivialDialect.td)
mlir_tablegen(TrivialDialect.h.inc -gen-dialect-decls -dialect=trivial)
mlir_tablegen(TrivialDialect.cpp.inc -gen-dialect-defs -dialect=trivial)
add_public_tablegen_target(TrivialDialectIncGen)

set(LLVM_TARGET_DEFINITIONS TrivialOps.td)
mlir_tablegen(TrivialOps.h.inc -gen-op-decls)
mlir_tablegen(TrivialOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TrivialOpsIncGen)

set(LLVM_TARGET_DEFINITIONS TrivialPasses.td)
mlir_tablegen(TrivialPasses.h.inc -gen-pass-decls -name Trivial)
add_public_tablegen_target(TrivialPassesIncGen)
