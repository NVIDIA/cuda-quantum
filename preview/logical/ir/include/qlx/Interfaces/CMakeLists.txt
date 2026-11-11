# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS LinearOwnershipInterfaces.td)
mlir_tablegen(LinearOwnershipInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(LinearOwnershipInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRQLXLinearOwnershipInterfacesIncGen)
