//===-- qlx-c/Passes.h - QLX pass registration --------------------*- C -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_C_PASSES_H
#define QLX_C_PASSES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Register every native QLX pass with the global pass registry.
MLIR_CAPI_EXPORTED void qlxRegisterAllPasses(void);

/// Run typed QLX/PBC transforms through the process-wide common C API image.
MLIR_CAPI_EXPORTED MlirLogicalResult
qlxVerifyCliffordTModule(MlirModule module);
MLIR_CAPI_EXPORTED MlirLogicalResult qlxLowerToPBCModule(MlirModule module);
MLIR_CAPI_EXPORTED MlirLogicalResult qlxVerifyPBCModule(MlirModule module);

#ifdef __cplusplus
}
#endif

#endif // QLX_C_PASSES_H
