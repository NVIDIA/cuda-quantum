/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// These loop normalization patterns are used by the cc-loop-normalize pass
// and cc-loop-unroll pass

// This file must be included after a `using namespace mlir;` as it uses bare
// identifiers from that namespace.

// Return true if \p loop is not monotonic or it is an invariant loop.
// Normalization is to be done on any loop that is monotonic and not invariant
// (which includes loops that are already in counted form).
static bool isNotMonotonicOrInvariant(cudaq::cc::LoopOp loop,
                                      bool allowClosedInterval,
                                      bool allowEarlyExit) {
  cudaq::opt::LoopComponents c;
  return !cudaq::opt::isaMonotonicLoop(loop, allowEarlyExit, &c) ||
         (cudaq::opt::isaInvariantLoop(c, allowClosedInterval) &&
          !c.isLinearExpr());
}

namespace {
class LoopPat : public OpRewritePattern<cudaq::cc::LoopOp> {
public:
  explicit LoopPat(MLIRContext *ctx, bool aci, bool ab)
      : OpRewritePattern(ctx), allowClosedInterval(aci), allowEarlyExit(ab) {}

  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loop,
                                PatternRewriter &rewriter) const override {
    if (loop->hasAttr(cudaq::opt::NormalizedLoopAttr) ||
        loop->hasAttr(cudaq::opt::DeadLoopAttr))
      return failure();
    if (isNotMonotonicOrInvariant(loop, allowClosedInterval, allowEarlyExit))
      return failure();

    // loop is monotonic but not invariant.
    LLVM_DEBUG(llvm::dbgs() << "loop before normalization: " << loop << '\n');
    auto componentsOpt = cudaq::opt::getLoopComponents(loop);
    assert(componentsOpt && "loop must have components");
    auto c = *componentsOpt;
    if (c.hasAlwaysTrueCondition()) {
      loop->emitWarning("Loop condition is always true. This loop is not "
                        "supported in a kernel.");
      return failure();
    }

    if (c.hasAlwaysFalseCondition()) {
      rewriter.startRootUpdate(loop);
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(c.compareOp, 0, 1);
      loop->setAttr(cudaq::opt::DeadLoopAttr, rewriter.getUnitAttr());
      rewriter.finalizeRootUpdate(loop);
      return success();
    }
    auto loc = loop.getLoc();

    // 1) Set initial value to 0.
    auto ty = c.initialValue.getType();
    rewriter.startRootUpdate(loop);
    auto createConstantOp = [&](std::int64_t val) -> Value {
      return rewriter.create<arith::ConstantIntOp>(loc, val, ty);
    };
    auto zero = createConstantOp(0);
    loop->setOperand(c.induction, zero);

    // 2) Compute the number of iterations as an invariant. `iterations = max(0,
    // (upper - lower + step) / step)`.
    Value upper = c.compareValue;
    auto one = createConstantOp(1);
    Value step = c.stepValue;
    Value lower = c.initialValue;
    if (!c.stepIsAnAddOp())
      step = rewriter.create<arith::SubIOp>(loc, zero, step);
    if (c.isLinearExpr()) {
      // Induction is part of a linear expression. Deal with the terms of the
      // equation. `m` scales the step. `b` is an addend to the lower bound.
      if (c.addendValue) {
        if (c.negatedAddend) {
          // `m * i - b`, u += `b`.
          upper = rewriter.create<arith::AddIOp>(loc, upper, c.addendValue);
        } else {
          // `m * i + b`, u -= `b`.
          upper = rewriter.create<arith::SubIOp>(loc, upper, c.addendValue);
        }
      }
      if (c.minusOneMult) {
        // `b - m * i` (b eliminated), multiply lower and step by `-1` (`m`
        // follows).
        auto negOne = createConstantOp(-1);
        lower = rewriter.create<arith::MulIOp>(loc, lower, negOne);
        step = rewriter.create<arith::MulIOp>(loc, step, negOne);
      }
      if (c.scaleValue) {
        if (c.reciprocalScale) {
          // `1/m * i + b` (b eliminated), multiply upper by `m`.
          upper = rewriter.create<arith::MulIOp>(loc, upper, c.scaleValue);
        } else {
          // `m * i + b` (b eliminated), multiple lower and step by `m`.
          lower = rewriter.create<arith::MulIOp>(loc, lower, c.scaleValue);
          step = rewriter.create<arith::MulIOp>(loc, step, c.scaleValue);
        }
      }
    }
    if (!c.isClosedIntervalForm()) {
      // Note: treating the step as a signed value to process countdown loops as
      // well as countup loops.
      Value negStepCond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, step, zero);
      auto negOne = createConstantOp(-1);
      Value adj =
          rewriter.create<arith::SelectOp>(loc, ty, negStepCond, negOne, one);
      upper = rewriter.create<arith::SubIOp>(loc, upper, adj);
    }
    Value diff = rewriter.create<arith::SubIOp>(loc, upper, lower);
    Value disp = rewriter.create<arith::AddIOp>(loc, diff, step);
    auto cmpOp = cast<arith::CmpIOp>(c.compareOp);
    Value newUpper = rewriter.create<arith::DivSIOp>(loc, disp, step);
    if (cudaq::opt::isSignedPredicate(cmpOp.getPredicate())) {
      Value noLoopCond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, newUpper, zero);
      newUpper =
          rewriter.create<arith::SelectOp>(loc, ty, noLoopCond, newUpper, zero);
    }

    // 3) Rewrite the comparison (!=) and step operations (+1).
    Value v1 = c.getCompareInduction();
    rewriter.setInsertionPoint(cmpOp);
    Value newCmp = rewriter.create<arith::CmpIOp>(
        cmpOp.getLoc(), arith::CmpIPredicate::ne, v1, newUpper);
    cmpOp->replaceAllUsesWith(ValueRange{newCmp});
    auto v2 = c.stepOp->getOperand(
        c.stepIsAnAddOp() && c.shouldCommuteStepOp() ? 1 : 0);
    rewriter.setInsertionPoint(c.stepOp);
    auto newStep = rewriter.create<arith::AddIOp>(c.stepOp->getLoc(), v2, one);
    c.stepOp->replaceAllUsesWith(ValueRange{newStep.getResult()});

    // 4) Compute original induction value as a loop variant and replace the
    // uses. `lower + step * i`. Careful to not replace the new induction.
    if (!loop.getBodyRegion().empty()) {
      Block *entry = &loop.getBodyRegion().front();
      rewriter.setInsertionPointToStart(entry);
      Value induct = entry->getArgument(c.induction);
      auto mul = rewriter.create<arith::MulIOp>(loc, induct, c.stepValue);
      Value newInd;
      if (c.stepIsAnAddOp())
        newInd = rewriter.create<arith::AddIOp>(loc, c.initialValue, mul);
      else
        newInd = rewriter.create<arith::SubIOp>(loc, c.initialValue, mul);
      induct.replaceUsesWithIf(newInd, [&](OpOperand &opnd) {
        auto *op = opnd.getOwner();
        return op != newStep.getOperation() && op != mul &&
               !isa<cudaq::cc::ContinueOp>(op);
      });
    }
    loop->setAttr(cudaq::opt::NormalizedLoopAttr, rewriter.getUnitAttr());

    rewriter.finalizeRootUpdate(loop);
    LLVM_DEBUG(llvm::dbgs() << "loop after normalization: " << loop << '\n');
    return success();
  }

  bool allowClosedInterval;
  bool allowEarlyExit;
};
} // namespace
