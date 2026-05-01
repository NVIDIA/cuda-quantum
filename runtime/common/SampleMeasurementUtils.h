/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/SampleMeasurementResolution.h"
#include "cudaq/Optimizer/Transforms/SampleMeasurementAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace cudaq::details {

inline void
resolveSampleExplicitMeasurementsFromFuncOp(mlir::func::FuncOp func,
                                            ExecutionContext &ctx,
                                            bool targetSupportsExplicit) {
  if (ctx.name != "sample")
    return;
  if (!func || !func->hasAttr("cudaq-kernel"))
    return;

  bool requiresExplicit = cudaq::opt::requiresExplicitMeasurements(func);
  resolveSampleExplicitMeasurements(ctx, requiresExplicit,
                                    targetSupportsExplicit);
}

} // namespace cudaq::details
