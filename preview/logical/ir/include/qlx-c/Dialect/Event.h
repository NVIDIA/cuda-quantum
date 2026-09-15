/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_C_DIALECT_EVENT_H
#define QLX_C_DIALECT_EVENT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Event, event);

//===----------------------------------------------------------------------===//
// !event.handle<payload, ownership, stream?>
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool eventTypeIsAHandle(MlirType type);

// `stream` may be a null MlirAttribute to construct a handle with no bound
// machine stream.
MLIR_CAPI_EXPORTED MlirType eventHandleTypeGet(MlirContext ctx,
                                               MlirType payload,
                                               MlirStringRef ownership,
                                               MlirAttribute stream);
MLIR_CAPI_EXPORTED MlirType eventHandleTypeGetPayload(MlirType type);
MLIR_CAPI_EXPORTED MlirStringRef eventHandleTypeGetOwnership(MlirType type);
// Returns a null MlirAttribute if the handle has no bound stream.
MLIR_CAPI_EXPORTED MlirAttribute eventHandleTypeGetStream(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID eventHandleTypeGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif
