# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS EventDialect.td)
mlir_tablegen(EventDialect.h.inc -gen-dialect-decls)
mlir_tablegen(EventDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIREventDialectIncGen)

set(LLVM_TARGET_DEFINITIONS EventTypes.td)
mlir_tablegen(EventTypes.h.inc -gen-typedef-decls)
mlir_tablegen(EventTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIREventTypesIncGen)

set(LLVM_TARGET_DEFINITIONS EventInterfaces.td)
mlir_tablegen(EventInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(EventInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIREventInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS EventOps.td)
mlir_tablegen(EventOps.h.inc -gen-op-decls)
mlir_tablegen(EventOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIREventOpsIncGen)
