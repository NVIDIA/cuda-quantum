# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name QLXConversion)
add_public_tablegen_target(MLIRQLXConversionPassesIncGen)

set(LLVM_TARGET_DEFINITIONS QuakeToQLXPasses.td)
mlir_tablegen(QuakeToQLXPasses.h.inc -gen-pass-decls -name QuakeToQLX)
add_public_tablegen_target(MLIRQuakeToQLXPassesIncGen)
