/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
    if (callee.equals(cudaq::stdMoveBuiltin)) {
      rewriter.replaceOp(call, call.getOperands());
      rewriter.eraseOp(call);
      return success();
    }
    return failure();
  }
};

class EraseNopCallsPass
    : public cudaq::opt::impl::EraseNopCallsBase<EraseNopCallsPass> {
public:
  using EraseNopCallsBase::EraseNopCallsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Function before erasure:\n" << func << "\n\n");
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<EraseStdMovePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "Function after erasure:\n" << func << "\n\n");
  }
};
} // namespace
