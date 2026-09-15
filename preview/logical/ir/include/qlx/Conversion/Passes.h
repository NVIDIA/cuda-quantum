/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// Mirrors upstream MLIR's `mlir/Conversion/Passes.h`.  Every cross-dialect
// QLX conversion pass declared in `qlx/Conversion/Passes.td` becomes
// available here as `qlx::create<Name>Pass()` and is registered by
// `qlx::registerQLXConversionPasses()`.
//
// Note: lives in the top-level `qlx` umbrella namespace alongside the
// project-owned MLIR dialect APIs (`qlx::QLXDialect`,
// `qlx::fabric::FabricDialect`, etc.).  Conversions cross dialects and
// don't belong to any single dialect sub-namespace.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_CONVERSION_PASSES_H
#define QLX_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace qlx {

#define GEN_PASS_DECL
#include "qlx/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "qlx/Conversion/Passes.h.inc"

} // namespace qlx

#endif // QLX_CONVERSION_PASSES_H
