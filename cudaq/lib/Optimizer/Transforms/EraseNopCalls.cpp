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
#define GEN_PASS_DEF_ERASENOPCALLS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "erase-nop-calls"

using namespace mlir;

namespace {
// Erase the std::move() call here.
class EraseStdMovePattern : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    auto callee = call.getCallee();
    if (callee == cudaq::stdMoveBuiltin) {
      rewriter.replaceOp(call, call.getOperands());
      rewriter.eraseOp(call);
      return success();
    }
    return failure();
  }
};

static bool isSampleOutputMarker(StringRef callee) {
  return callee == cudaq::sampleOutputQubitMarker ||
         callee == cudaq::sampleOutputVeqMarker;
}

class EraseSampleOutputCallPattern : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (!isSampleOutputMarker(call.getCallee()) || call.getNumResults())
      return failure();
    rewriter.eraseOp(call);
    return success();
  }
};

class EraseSampleOutputCallByRefPattern
    : public OpRewritePattern<cudaq::quake::CallByRefOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::CallByRefOp call,
                                PatternRewriter &rewriter) const override {
    if (!isSampleOutputMarker(call.getCallee()))
      return failure();
    if (call.getNumResults() != call.getNumOperands() ||
        call.getResultTypes() != call.getOperandTypes())
      return failure();
    rewriter.replaceOp(call, call.getOperands());
    return success();
  }
};

class EraseNopCallsPass
    : public cudaq::opt::impl::EraseNopCallsBase<EraseNopCallsPass> {
public:
  using EraseNopCallsBase::EraseNopCallsBase;

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before erasure:\n" << *op << "\n\n");
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<EraseSampleOutputCallByRefPattern,
                    EraseSampleOutputCallPattern, EraseStdMovePattern>(ctx);
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After erasure:\n" << *op << "\n\n");
  }
};
} // namespace
