# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS SemanticInterfaces.td)
mlir_tablegen(SemanticInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SemanticInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRCUDAQLogicalSemanticInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS PhysicalInterfaces.td)
mlir_tablegen(PhysicalInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(PhysicalInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRCUDAQLogicalPhysicalInterfacesIncGen)

add_custom_target(MLIRCUDAQLogicalInterfacesIncGen)
add_dependencies(MLIRCUDAQLogicalInterfacesIncGen
  MLIRCUDAQLogicalSemanticInterfacesIncGen
  MLIRCUDAQLogicalPhysicalInterfacesIncGen
)
