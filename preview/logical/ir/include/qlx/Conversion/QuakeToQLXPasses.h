/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_CONVERSION_QUAKETOQLX_PASSES_H
#define QLX_CONVERSION_QUAKETOQLX_PASSES_H

#include "mlir/Pass/Pass.h"

namespace qlx {

#define GEN_PASS_DECL
#include "qlx/Conversion/QuakeToQLXPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "qlx/Conversion/QuakeToQLXPasses.h.inc"

/// Register the `prepare-quake-for-qlx` named MLIR pipeline. Its input must
/// already have passed through CUDA-Q's `convert-to-linear-values` preparation;
/// the pipeline owns only the ordered QLX-side boundary cleanup required before
/// typed P0 conversion.
void registerPrepareQuakeForQLXPipeline();

} // namespace qlx

#endif // QLX_CONVERSION_QUAKETOQLX_PASSES_H
