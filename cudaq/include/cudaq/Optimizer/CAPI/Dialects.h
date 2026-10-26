/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Quake, quake);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QEC, qec);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(CC, cc);

// Register Quake, CC, and relevant upstream MLIR dialects into `registry`.
MLIR_CAPI_EXPORTED void cudaqRegisterAllDialects(MlirDialectRegistry registry);

// Load Quake, CC, and relevant upstream MLIR dialects into `context`.
MLIR_CAPI_EXPORTED void cudaqLoadAllDialects(MlirContext context);

#ifdef __cplusplus
}
#endif
