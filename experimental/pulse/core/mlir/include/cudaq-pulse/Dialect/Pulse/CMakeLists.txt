# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS PulseOps.td)
mlir_tablegen(PulseOps.h.inc -gen-op-decls)
mlir_tablegen(PulseOps.cpp.inc -gen-op-defs)
mlir_tablegen(PulseDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PulseDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS PulseTypes.td)
mlir_tablegen(PulseTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PulseTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(PulseEnums.h.inc -gen-enum-decls)
mlir_tablegen(PulseEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(CudaqPulsePulseIncGen)
