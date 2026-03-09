/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ERASENOISE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "erase-noise"

using namespace mlir;

/// \file
/// This pass exists simply to remove all the quake.apply_noise Ops from the IR.

namespace {
class EraseApplyNoisePattern : public OpRewritePattern<quake::ApplyNoiseOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ApplyNoiseOp noise,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(noise);
    return success();
  }
};

class EraseNoisePass : public cudaq::opt::impl::EraseNoiseBase<EraseNoisePass> {
public:
  using EraseNoiseBase::EraseNoiseBase;

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before erasure:\n" << *op << "\n\n");
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<EraseApplyNoisePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After erasure:\n" << *op << "\n\n");
  }
};
} // namespace
