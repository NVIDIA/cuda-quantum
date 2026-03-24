/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
      rewriter.startOpModification(loop);
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(c.compareOp, 0, 1);
      loop->setAttr(cudaq::opt::DeadLoopAttr, rewriter.getUnitAttr());
      rewriter.finalizeOpModification(loop);
      return success();
    }
    auto loc = loop.getLoc();

    // 1) Set initial value to 0.
    auto ty = c.initialValue.getType();
    rewriter.startOpModification(loop);
    auto createConstantOp = [&](std::int64_t val) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, ty, val);
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
      step = arith::SubIOp::create(rewriter, loc, zero, step);
    if (c.isLinearExpr()) {
      // Induction is part of a linear expression. Deal with the terms of the
      // equation. `m` scales the step. `b` is an addend to the lower bound.
      if (c.addendValue) {
        if (c.negatedAddend) {
          // `m * i - b`, u += `b`.
          upper = arith::AddIOp::create(rewriter, loc, upper, c.addendValue);
        } else {
          // `m * i + b`, u -= `b`.
          upper = arith::SubIOp::create(rewriter, loc, upper, c.addendValue);
        }
      }
      if (c.minusOneMult) {
        // `b - m * i` (b eliminated), multiply lower and step by `-1` (`m`
        // follows).
        auto negOne = createConstantOp(-1);
        lower = arith::MulIOp::create(rewriter, loc, lower, negOne);
        step = arith::MulIOp::create(rewriter, loc, step, negOne);
      }
      if (c.scaleValue) {
        if (c.reciprocalScale) {
          // `1/m * i + b` (b eliminated), multiply upper by `m`.
          upper = arith::MulIOp::create(rewriter, loc, upper, c.scaleValue);
        } else {
          // `m * i + b` (b eliminated), multiple lower and step by `m`.
          lower = arith::MulIOp::create(rewriter, loc, lower, c.scaleValue);
          step = arith::MulIOp::create(rewriter, loc, step, c.scaleValue);
        }
      }
    }
    if (!c.isClosedIntervalForm()) {
      // Note: treating the step as a signed value to process countdown loops as
      // well as countup loops.
      Value negStepCond = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, step, zero);
      auto negOne = createConstantOp(-1);
      Value adj =
          arith::SelectOp::create(rewriter, loc, ty, negStepCond, negOne, one);
      upper = arith::SubIOp::create(rewriter, loc, upper, adj);
    }
    Value diff = arith::SubIOp::create(rewriter, loc, upper, lower);
    Value disp = arith::AddIOp::create(rewriter, loc, diff, step);
    auto cmpOp = cast<arith::CmpIOp>(c.compareOp);
    Value newUpper = arith::DivSIOp::create(rewriter, loc, disp, step);
    if (cudaq::opt::isSignedPredicate(cmpOp.getPredicate())) {
      Value noLoopCond = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sgt, newUpper, zero);
      newUpper =
          arith::SelectOp::create(rewriter, loc, ty, noLoopCond, newUpper, zero);
    }

    // 3) Rewrite the comparison (!=) and step operations (+1).
    Value v1 = c.getCompareInduction();
    rewriter.setInsertionPoint(cmpOp);
    Value newCmp = arith::CmpIOp::create(
        rewriter, cmpOp.getLoc(), arith::CmpIPredicate::ne, v1, newUpper);
    cmpOp->replaceAllUsesWith(ValueRange{newCmp});
    auto v2 = c.stepOp->getOperand(
        c.stepIsAnAddOp() && c.shouldCommuteStepOp() ? 1 : 0);
    rewriter.setInsertionPoint(c.stepOp);
    auto newStep = arith::AddIOp::create(rewriter, c.stepOp->getLoc(), v2, one);
    c.stepOp->replaceAllUsesWith(ValueRange{newStep.getResult()});

    // 4) Compute original induction value as a loop variant and replace the
    // uses. `lower + step * i`. Careful to not replace the new induction.
    if (!loop.getBodyRegion().empty()) {
      Block *entry = &loop.getBodyRegion().front();
      rewriter.setInsertionPointToStart(entry);
      Value induct = entry->getArgument(c.induction);
      auto mul = arith::MulIOp::create(rewriter, loc, induct, c.stepValue);
      Value newInd;
      if (c.stepIsAnAddOp())
        newInd = arith::AddIOp::create(rewriter, loc, c.initialValue, mul);
      else
        newInd = arith::SubIOp::create(rewriter, loc, c.initialValue, mul);
      induct.replaceUsesWithIf(newInd, [&](OpOperand &opnd) {
        auto *op = opnd.getOwner();
        return op != newStep.getOperation() && op != mul &&
               !isa<cudaq::cc::ContinueOp>(op);
      });
    }
    loop->setAttr(cudaq::opt::NormalizedLoopAttr, rewriter.getUnitAttr());

    rewriter.finalizeOpModification(loop);
    LLVM_DEBUG(llvm::dbgs() << "loop after normalization: " << loop << '\n');
    return success();
  }

  bool allowClosedInterval;
  bool allowEarlyExit;
};
} // namespace
