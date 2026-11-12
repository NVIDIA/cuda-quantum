/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Analysis/LinearOwnership.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Optimizer/Transforms/VerifyLinearOwnership.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_VERIFYLINEAROWNERSHIP
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {
class VerifyLinearOwnershipPass
    : public cudaq::opt::impl::VerifyLinearOwnershipBase<
          VerifyLinearOwnershipPass> {
public:
  using VerifyLinearOwnershipBase::VerifyLinearOwnershipBase;

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

void cudaq::opt::registerLinearOwnershipVerifierPass() {
  registerVerifyLinearOwnership();
}
