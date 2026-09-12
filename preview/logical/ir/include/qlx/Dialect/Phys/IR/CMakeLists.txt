# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS PhysDialect.td)
mlir_tablegen(PhysDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PhysDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRPhysDialectIncGen)

set(LLVM_TARGET_DEFINITIONS PhysTypes.td)
mlir_tablegen(PhysTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PhysTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRPhysTypesIncGen)

set(LLVM_TARGET_DEFINITIONS PhysOps.td)
mlir_tablegen(PhysOps.h.inc -gen-op-decls)
mlir_tablegen(PhysOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRPhysOpsIncGen)
