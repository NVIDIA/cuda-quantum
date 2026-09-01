//===-- qlx-c/Translate/Translations.h - C API for QLX translations -*- C
//-*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// C interface for QLX-side MLIR translations:
//   - fabric-to-stim  (fabric circuit -> Stim text)
//
// Each translation streams its result through an `MlirStringCallback` to
// avoid allocating a single contiguous buffer in the C ABI.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_C_TRANSLATE_TRANSLATIONS_H
#define QLX_C_TRANSLATE_TRANSLATIONS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Translate a verified Fabric module to standards-compatible Stim text.
/// Unsupported operations fail closed.
MLIR_CAPI_EXPORTED MlirLogicalResult qlxTranslateFabricToStim(
    MlirModule module, MlirStringCallback callback, void *userData);

#ifdef __cplusplus
}
#endif

#endif // QLX_C_TRANSLATE_TRANSLATIONS_H
