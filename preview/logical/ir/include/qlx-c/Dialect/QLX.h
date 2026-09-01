/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef QLX_C_DIALECT_QLX_H
#define QLX_C_DIALECT_QLX_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(QLX, qlx);

MLIR_CAPI_EXPORTED bool qlxLowerToPBC(MlirModule module);

MLIR_CAPI_EXPORTED bool qlxAttributeIsAPauli(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute qlxPauliAttrGet(MlirContext ctx,
                                                 MlirStringRef value);
MLIR_CAPI_EXPORTED MlirStringRef qlxPauliAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID qlxPauliAttrGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // QLX_C_DIALECT_QLX_H
