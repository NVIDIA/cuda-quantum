/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qlx/Dialect/QLX/Transforms/Passes.h"
#include "cudaq/Optimizer/Analysis/LinearOwnership.h"
#include "mlir/IR/BuiltinOps.h"

namespace qlx {
#define GEN_PASS_DEF_QLXVERIFYLINEARITY
#include "qlx/Dialect/QLX/Transforms/Passes.h.inc"
} // namespace qlx

namespace {

struct QLXVerifyLinearityPass
    : public qlx::impl::QLXVerifyLinearityBase<QLXVerifyLinearityPass> {
  void runOnOperation() override {
    cudaq::opt::LinearOwnershipReport report =
        cudaq::opt::analyzeLinearOwnership(getOperation());
    for (const cudaq::opt::LinearOwnershipViolation &violation :
         report.violations)
      cudaq::opt::emitLinearOwnershipViolation(violation);
    if (!report.ok())
      signalPassFailure();
    markAllAnalysesPreserved();
  }
};

} // namespace
