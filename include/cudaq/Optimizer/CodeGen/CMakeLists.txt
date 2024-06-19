# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_cudaq_dialect(CodeGen codegen)
add_cudaq_dialect_doc(CodeGenDialect codegen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name OptCodeGen)
add_public_tablegen_target(OptCodeGenPassIncGen)

set(LLVM_TARGET_DEFINITIONS Peephole.td)
mlir_tablegen(Peephole.inc -gen-rewriters)
add_public_tablegen_target(OptPeepholeIncGen)
