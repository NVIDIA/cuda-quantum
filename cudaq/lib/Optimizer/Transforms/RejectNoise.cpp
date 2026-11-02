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
#define GEN_PASS_DEF_REJECTNOISE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {
class RejectNoisePass
    : public cudaq::opt::impl::RejectNoiseBase<RejectNoisePass> {
public:
  using RejectNoiseBase::RejectNoiseBase;

  void runOnOperation() override {
    bool hasNoise = false;
    getOperation()->walk([&](cudaq::quake::ApplyNoiseOp noise) {
      noise.emitError("the target does not support noise");
      hasNoise = true;
    });
    if (hasNoise)
      signalPassFailure();
  }
};
} // namespace
