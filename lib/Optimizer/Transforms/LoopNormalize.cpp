/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOOPNORMALIZE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "cc-loop-normalize"

using namespace mlir;

#include "LoopNormalizePatterns.inc"

namespace {
class LoopNormalizePass
    : public cudaq::opt::impl::LoopNormalizeBase<LoopNormalizePass> {
public:
  using LoopNormalizeBase::LoopNormalizeBase;

  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<LoopPat>(ctx, allowClosedInterval, allowBreak);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      op->emitOpError("could not normalize loop");
      signalPassFailure();
    }
  }
};
} // namespace
