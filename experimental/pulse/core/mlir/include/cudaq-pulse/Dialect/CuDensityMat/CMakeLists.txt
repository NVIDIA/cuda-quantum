# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS CuDensityMatOps.td)
mlir_tablegen(CuDensityMatOps.h.inc -gen-op-decls)
mlir_tablegen(CuDensityMatOps.cpp.inc -gen-op-defs)
mlir_tablegen(CuDensityMatDialect.h.inc -gen-dialect-decls)
mlir_tablegen(CuDensityMatDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS CuDensityMatTypes.td)
mlir_tablegen(CuDensityMatTypes.h.inc -gen-typedef-decls)
mlir_tablegen(CuDensityMatTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS CuDensityMatAttrs.td)
mlir_tablegen(CuDensityMatAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(CuDensityMatAttrs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(CuDensityMatEnums.h.inc -gen-enum-decls)
mlir_tablegen(CuDensityMatEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(CudaqPulseCuDensityMatIncGen)
