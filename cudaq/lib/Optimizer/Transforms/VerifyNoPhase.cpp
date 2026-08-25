/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_VERIFYNOPHASE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {

struct VerifyNoPhasePass
    : public cudaq::opt::impl::VerifyNoPhaseBase<VerifyNoPhasePass> {
  using VerifyNoPhaseBase::VerifyNoPhaseBase;

  void runOnOperation() override {
    bool foundPhase = false;

    getOperation().walk([&](cudaq::quake::PhaseOp phase) {
      phase.emitOpError(
          "phase bookkeeping must be lowered before code generation");
      foundPhase = true;
    });

    if (foundPhase)
      signalPassFailure();
  }
};

} // namespace
