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
    if (!c.induction.has_value())
      return failure();

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
    loop->setOperand(*c.induction, zero);

    // 2) Compute the number of iterations as an invariant. `iterations = max(0,
    // (upper - lower + step) / step)`.
    // isLoopInvariant (which getLoopComponents relies on) accepts values
    // that are loop-invariant in *value* but are defined by a pure
    // recomputation nested inside the loop (e.g. `quake.veq_size` on a veq
    // the loop doesn't modify), not just values that already dominate the
    // loop. Materialize (hoist a clone of) any such value here so it can
    // legally be used at this insertion point, which must dominate the loop.
    Value upper = cudaq::opt::materializeLoopInvariant(rewriter,
                                                       c.compareValue, loop);
    auto one = createConstantOp(1);
    Value step =
        cudaq::opt::materializeLoopInvariant(rewriter, c.stepValue, loop);
    Value lower =
        cudaq::opt::materializeLoopInvariant(rewriter, c.initialValue, loop);

    // upper, step, and lower are artifacts pulled from the original
    // function (e.g. the loop's own bound vs. its induction/step) and are
    // not guaranteed to share an integer width: the bound may be computed
    // in a wider type than the induction was declared with (as happens
    // when the bound is compared against a widened copy of the
    // induction, such as a veq size). Do all the arithmetic below in a
    // single, common (widest) integer type, promoting narrower values up
    // via a signed cc.cast; `newUpper` is narrowed back down to `ty` (the
    // induction's own type) just before it replaces the comparison below.
    auto widerOf = [](Type a, Type b) {
      return cast<IntegerType>(a).getWidth() >= cast<IntegerType>(b).getWidth()
                 ? a
                 : b;
    };
    Type wideTy =
        widerOf(widerOf(upper.getType(), step.getType()), lower.getType());
    auto promote = [&](Value v) -> Value {
      if (v.getType() == wideTy)
        return v;
      return cudaq::cc::CastOp::create(rewriter, loc, wideTy, v,
                                       cudaq::cc::CastOpMode::Signed);
    };
    upper = promote(upper);
    step = promote(step);
    lower = promote(lower);
    auto wideZero = wideTy == ty ? zero : promote(zero);
    auto wideOne = wideTy == ty ? one : promote(one);

    if (!c.stepIsAnAddOp())
      step = arith::SubIOp::create(rewriter, loc, wideZero, step);
    if (c.isLinearExpr()) {
      // Induction is part of a linear expression. Deal with the terms of the
      // equation. `m` scales the step. `b` is an addend to the lower bound.
      if (c.addendValue) {
        Value addend = promote(cudaq::opt::materializeLoopInvariant(
            rewriter, c.addendValue, loop));
        if (c.negatedAddend) {
          // `m * i - b`, u += `b`.
          upper = arith::AddIOp::create(rewriter, loc, upper, addend);
        } else {
          // `m * i + b`, u -= `b`.
          upper = arith::SubIOp::create(rewriter, loc, upper, addend);
        }
      }
      if (c.minusOneMult) {
        // `b - m * i` (b eliminated), multiply lower and step by `-1` (`m`
        // follows).
        auto wideNegOne = arith::ConstantIntOp::create(rewriter, loc, wideTy, -1);
        lower = arith::MulIOp::create(rewriter, loc, lower, wideNegOne);
        step = arith::MulIOp::create(rewriter, loc, step, wideNegOne);
      }
      if (c.scaleValue) {
        Value scale = promote(cudaq::opt::materializeLoopInvariant(
            rewriter, c.scaleValue, loop));
        if (c.reciprocalScale) {
          // `1/m * i + b` (b eliminated), multiply upper by `m`.
          upper = arith::MulIOp::create(rewriter, loc, upper, scale);
        } else {
          // `m * i + b` (b eliminated), multiple lower and step by `m`.
          lower = arith::MulIOp::create(rewriter, loc, lower, scale);
          step = arith::MulIOp::create(rewriter, loc, step, scale);
        }
      }
    }
    if (!c.isClosedIntervalForm()) {
      // Note: treating the step as a signed value to process countdown loops as
      // well as countup loops.
      Value negStepCond = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, step, wideZero);
      auto wideNegOne = arith::ConstantIntOp::create(rewriter, loc, wideTy, -1);
      Value adj = arith::SelectOp::create(rewriter, loc, wideTy, negStepCond,
                                          wideNegOne, wideOne);
      upper = arith::SubIOp::create(rewriter, loc, upper, adj);
    }
    Value diff = arith::SubIOp::create(rewriter, loc, upper, lower);
    Value disp = arith::AddIOp::create(rewriter, loc, diff, step);
    auto cmpOp = cast<arith::CmpIOp>(c.compareOp);
    Value newUpper = arith::DivSIOp::create(rewriter, loc, disp, step);
    if (cudaq::opt::isSignedPredicate(cmpOp.getPredicate())) {
      Value noLoopCond = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::sgt, newUpper, wideZero);
      newUpper = arith::SelectOp::create(rewriter, loc, wideTy, noLoopCond,
                                         newUpper, wideZero);
    }
    // 3) Rewrite the comparison (!=) and step operations (+1).
    // getCompareInduction() returns the *other* operand of the original
    // comparison (i.e. whatever `compareValue`, now folded into newUpper,
    // was being compared against). That operand may itself be a widened
    // (cast) copy of the induction rather than the induction's own `ty`, so
    // promote whichever of v1/newUpper is narrower to match the other
    // instead of assuming either one's type.
    Value v1 = c.getCompareInduction();
    if (v1.getType() != newUpper.getType()) {
      if (widerOf(v1.getType(), newUpper.getType()) == v1.getType())
        newUpper = promote(newUpper);
      else
        v1 = cudaq::cc::CastOp::create(rewriter, loc, newUpper.getType(), v1,
                                       cudaq::cc::CastOpMode::Signed);
    }
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
      Value induct = entry->getArgument(*c.induction);
      auto mul = arith::MulIOp::create(rewriter, loc, induct, c.stepValue);
      auto newInd = [&]() -> Value {
        if (c.stepIsAnAddOp())
          return arith::AddIOp::create(rewriter, loc, c.initialValue, mul);
        return arith::SubIOp::create(rewriter, loc, c.initialValue, mul);
      }();
      induct.replaceUsesWithIf(newInd, [&](OpOperand &opnd) {
        auto *op = opnd.getOwner();
        return op != newStep.getOperation() && op != mul &&
               !isa<cudaq::cc::ContinueOp>(op);
      });
    }

    // 5) Back-convert the loop's induction-position result for external uses
    if (*c.induction < loop.getNumResults()) {
      Value loopResult = loop.getResult(*c.induction);
      if (!loopResult.use_empty()) {
        rewriter.setInsertionPointAfter(loop);
        auto mulRes =
            arith::MulIOp::create(rewriter, loc, loopResult, c.stepValue);
        Value recovered;
        if (c.stepIsAnAddOp())
          recovered =
              arith::AddIOp::create(rewriter, loc, c.initialValue, mulRes);
        else
          recovered =
              arith::SubIOp::create(rewriter, loc, c.initialValue, mulRes);
        loopResult.replaceAllUsesExcept(recovered, mulRes.getOperation());
      }
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
