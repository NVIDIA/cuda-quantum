/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "cudaq/Optimizer/Transforms/SampleMeasurementAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <stdexcept>

namespace cudaq::details {

inline void resolveSampleExplicitMeasurements(mlir::func::FuncOp func,
                                              ExecutionContext &ctx,
                                              bool targetSupportsExplicit) {
  if (ctx.name != "sample")
    return;
  if (!func || !func->hasAttr("cudaq-kernel"))
    return;

  if (!cudaq::opt::requiresExplicitMeasurements(func)) {
    ctx.explicitMeasurements = false;
    return;
  }

  if (!targetSupportsExplicit)
    throw std::runtime_error(
        "This kernel requires explicit measurement result semantics for "
        "sampling, but explicit measurements are not supported on this "
        "target.");

  // if the analysis requires explicit measurements and the
  // target supports them, enable them regardless of the initial context
  // value. The initial value may be false because kernel metadata was not
  // yet available when the default was computed (e.g. template kernels
  // whose Quake IR is registered during JIT compilation).
  ctx.explicitMeasurements = true;
}

} // namespace cudaq::details
