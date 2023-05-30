/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "raise-to-affine"

using namespace mlir;

namespace {
class RewriteLoop : public OpRewritePattern<cudaq::cc::LoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    TODO("loop lowering");
    return success();
  }
};

class RaiseToAffinePass
    : public cudaq::opt::RaiseToAffineBase<RaiseToAffinePass> {
public:
  RaiseToAffinePass() = default;

  void runOnOperation() override {
    auto op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<RewriteLoop>(ctx);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<cudaq::cc::ScopeOp>(
        [](cudaq::cc::ScopeOp x) { return true; });
    target.addDynamicallyLegalOp<cudaq::cc::LoopOp>(
        [](cudaq::cc::LoopOp x) { return true; });
    target.addDynamicallyLegalOp<cudaq::cc::IfOp>(
        [](cudaq::cc::IfOp x) { return true; });
    target.addDynamicallyLegalOp<cudaq::cc::ConditionOp>(
        [](cudaq::cc::ConditionOp x) { return true; });
    target.addDynamicallyLegalOp<cudaq::cc::ContinueOp>(
        [](cudaq::cc::ContinueOp x) { return true; });
    target.addDynamicallyLegalOp<cudaq::cc::BreakOp>(
        [](cudaq::cc::BreakOp x) { return true; });
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createRaiseToAffinePass() {
  return std::make_unique<RaiseToAffinePass>();
}
