/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOOPNORMALIZE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "cc-loop-normalize"

using namespace mlir;

namespace {
class LoopPat : public OpRewritePattern<cudaq::cc::LoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loop,
                                PatternRewriter &rewriter) const override {
    // loop is monotonic but not invariant.
    LLVM_DEBUG(llvm::dbgs() << "loop before normalization: " << loop << '\n');
    auto componentsOpt = cudaq::opt::getLoopComponents(loop);
    assert(componentsOpt && "loop must have components");
    auto c = *componentsOpt;
    auto loc = loop.getLoc();

    // 1) Set initial value to 0.
    auto ty = c.initialValue.getType();
    rewriter.startRootUpdate(loop);
    auto zero = rewriter.create<arith::ConstantIntOp>(loc, 0, ty);
    loop->setOperand(c.induction, zero);

    // 2) Compute the number of iterations as an invariant. `iterations = max(0,
    // (upper - lower + step) / step)`.
    Value upper = c.compareValue;
    auto one = rewriter.create<arith::ConstantIntOp>(loc, 1, ty);
    Value step = c.stepValue;
    if (!c.stepIsAnAddOp())
      step = rewriter.create<arith::SubIOp>(loc, zero, step);
    if (!c.isClosedIntervalForm()) {
      // Note: treating the step as a signed value to process countdown loops as
      // well as countup loops.
      Value negStepCond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, step, zero);
      auto negOne = rewriter.create<arith::ConstantIntOp>(loc, -1, ty);
      Value adj =
          rewriter.create<arith::SelectOp>(loc, ty, negStepCond, negOne, one);
      upper = rewriter.create<arith::SubIOp>(loc, upper, adj);
    }
    Value diff = rewriter.create<arith::SubIOp>(loc, upper, c.initialValue);
    Value disp = rewriter.create<arith::AddIOp>(loc, diff, step);
    auto cmpOp = cast<arith::CmpIOp>(c.compareOp);
    Value up1 = rewriter.create<arith::DivSIOp>(loc, disp, step);
    Value noLoopCond = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, up1, zero);
    Value newUpper =
        rewriter.create<arith::SelectOp>(loc, ty, noLoopCond, up1, zero);

    // 3) Rewrite the comparison (!=) and step operations (+1).
    Value v1 =
        cmpOp.getLhs() == c.compareValue ? cmpOp.getRhs() : cmpOp.getLhs();
    rewriter.setInsertionPoint(cmpOp);
    Value newCmp = rewriter.create<arith::CmpIOp>(
        cmpOp.getLoc(), arith::CmpIPredicate::ne, v1, newUpper);
    cmpOp->replaceAllUsesWith(ValueRange{newCmp});
    auto v2 = c.stepOp->getOperand(
        c.stepIsAnAddOp() && c.shouldCommuteStepOp() ? 1 : 0);
    rewriter.setInsertionPoint(c.stepOp);
    Value newStep = rewriter.create<arith::AddIOp>(c.stepOp->getLoc(), v2, one);
    c.stepOp->replaceAllUsesWith(ValueRange{newStep});

    // 4) Compute original induction value as a loop variant and replace the
    // uses. `lower + step * i`. Careful to not replace the new induction.
    if (!loop.getBodyRegion().empty()) {
      Block *entry = &loop.getBodyRegion().front();
      rewriter.setInsertionPointToStart(entry);
      Value induct = entry->getArgument(c.induction);
      auto mul = rewriter.create<arith::MulIOp>(loc, induct, c.stepValue);
      Value newInd = rewriter.create<arith::AddIOp>(loc, mul, c.initialValue);
      induct.replaceUsesWithIf(newInd, [&](OpOperand &opnd) {
        auto *op = opnd.getOwner();
        return op != mul && !isa<cudaq::cc::ContinueOp>(op);
      });
    }

    rewriter.finalizeRootUpdate(loop);
    LLVM_DEBUG(llvm::dbgs() << "loop after normalization: " << loop << '\n');
    return success();
  }
};

class LoopNormalizePass
    : public cudaq::opt::impl::LoopNormalizeBase<LoopNormalizePass> {
public:
  using LoopNormalizeBase::LoopNormalizeBase;

  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<LoopPat>(ctx);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<cudaq::cc::LoopOp>(
        [&](cudaq::cc::LoopOp loop) {
          cudaq::opt::LoopComponents c;
          return !cudaq::opt::isaMonotonicLoop(loop, &c) ||
                 cudaq::opt::isaInvariantLoop(c, /*allowClosedInterval=*/true);
        });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      op->emitOpError("could not normalize loop");
      signalPassFailure();
    }
  }
};
} // namespace
