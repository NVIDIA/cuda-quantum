/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// C interface for QLX-side MLIR translations:
//   - fabric-to-stim  (fabric circuit -> Stim text)
//
// Text translations and projections stream through an MlirStringCallback.
// EDEM derivation returns an owned packed buffer through the C ownership
// record.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_C_TRANSLATE_TRANSLATIONS_H
#define QLX_C_TRANSLATE_TRANSLATIONS_H

#include "cudaq/gf2/memory.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Translate a verified Fabric module to standards-compatible Stim text.
/// Unsupported operations fail closed.
MLIR_CAPI_EXPORTED MlirLogicalResult qlxTranslateFabricToStim(
    MlirModule module, MlirStringCallback callback, void *userData);

/// Project a selected gadget without modifying the module. On success, emits
/// a cudaq.logical.clifford-projection/v1 JSON value snapshot through callback.
/// The shared CAPI library owns the Fabric traversal and dialect identities.
/// An empty gadget name selects the unique entry gadget.
MLIR_CAPI_EXPORTED MlirLogicalResult
qlxProjectCliffordJson(MlirModule module, MlirStringRef gadget,
                       MlirStringCallback callback, void *userData);

/// Returns one owned, checked packed EDEM without copying its payload.
/// An empty profile selects the unique fabric.gadget_contract. Faults use the
/// external cudaq.logical.edem-faults/v1 JSON schema. Does not mutate the
/// module. Output must point to a zero-initialized record. Failure leaves it
/// unchanged; release the returned record with
/// cudaq_gf2_owned_buffer_release. Checked view tokens do not cross this
/// boundary.
MLIR_CAPI_EXPORTED MlirLogicalResult
qlxDeriveGadgetEDEM(MlirModule module, MlirStringRef profile,
                    MlirStringRef faults, cudaq_gf2_owned_buffer *output);

#ifdef __cplusplus
}
#endif

#endif // QLX_C_TRANSLATE_TRANSLATIONS_H
