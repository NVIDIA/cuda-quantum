# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name OptTransforms)
add_public_tablegen_target(OptTransformsPassIncGen)

add_cudaq_doc(Passes Transforms -gen-pass-doc)

# Passes.h includes the generated declarations, so downstream tools that
# register or create CUDA-Q passes need the .inc installed next to it.
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Passes.h.inc
        DESTINATION include/cudaq/Optimizer/Transforms
        COMPONENT Development)
