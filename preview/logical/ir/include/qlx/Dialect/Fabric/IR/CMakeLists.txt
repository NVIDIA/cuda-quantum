# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS FabricDialect.td)
mlir_tablegen(FabricDialect.h.inc -gen-dialect-decls)
mlir_tablegen(FabricDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRFabricDialectIncGen)

set(LLVM_TARGET_DEFINITIONS FabricInterfaces.td)
mlir_tablegen(FabricInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(FabricInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRFabricInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS FabricTypes.td)
mlir_tablegen(FabricTypes.h.inc -gen-typedef-decls)
mlir_tablegen(FabricTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRFabricTypesIncGen)

set(LLVM_TARGET_DEFINITIONS FabricAttrs.td)
mlir_tablegen(FabricAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(FabricAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRFabricAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS FabricAttrs.td)
mlir_tablegen(FabricAttrEnums.h.inc -gen-enum-decls)
mlir_tablegen(FabricAttrEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRFabricAttrEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS FabricOps.td)
mlir_tablegen(FabricOps.h.inc -gen-op-decls)
mlir_tablegen(FabricOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRFabricOpsIncGen)
