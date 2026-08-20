/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;

/// \file
/// Some working definitions:
///
/// A \em counted loop: a loop that counts from $0$ up to $n-1$ stepping by $1$.
/// Such a loop is \em normalized (starts at $0$), \em monotonically increasing
/// (slope is a constant $1$), executes exactly $n$ times, and $n$ is a
/// compile-time constant. A counted loop is said to have static control flow.
///
/// A \em{constant upper bound} loop: a loop that counts from $0$ up to $m$
/// where $m <= n-1$ stepping by $1$.
///
/// An \em invariant loop: a counted loop but $n$ need not be a compile-time
/// constant. An invariant loop cannot be fully unrolled until runtime. In
/// quantum circuit speak, one does not know the full size of the circuit.
///
/// A \em monotonic loop: a loop that counts from $i$ up to (down to) $j$
/// stepping by positive (negative) integral values; mathematically, it is a
/// strictly monotonic sequence. If the step is a compile-time constant, $k$,
/// then a closed iterval definite monotonic loop must execute exactly $\max(0,
/// \floor{(j - i + k) / k})$ iterations. By normalizing a monotonic loop and
/// constant folding and propagation, we may be able to convert it to static
/// control flow.
///
/// For completeness, a \em{conditionally iterated} loop is a monotonic loop
/// that has a second auxilliary condition to determine if a given loop
/// iteration is executed or not. (A constant upper bound loop, see above, is a
/// subclass of a conditionally iterated loop.) For example, the condition might
/// be used in iteration $m$ to disable all subsequent iterations. (Much like a
/// `break` statement.) Another example would be a condition that disables all
/// the even iterations. These loops might be unrolled but only if the loop can
/// be normalized into static control flow. It is helpful in pruning the amount
/// of unrolling if the auxillary condition can be computed as a constant. It is
/// likely these loops cannot be converted to static control flow and would thus
/// need to be expanded at runtime.

static Value peelCastOps(Value v) {
  Operation *defOp = nullptr;
  for (; (defOp = v.getDefiningOp());) {
    if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp,
            cudaq::cc::CastOp>(defOp))
      v = defOp->getOperand(0);
    else
      break;
  }
  return v;
}

static bool isaConstant(Value v) {
  v = peelCastOps(v);
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    return isa<IntegerAttr>(c.getValue());
  return false;
}

static bool isaConstantOf(Value v, std::int64_t hasVal) {
  v = peelCastOps(v);
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt() == hasVal;
  return false;
}

static bool isClosedIntervalForm(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::ule || p == arith::CmpIPredicate::sle;
}

static bool isClosedIntervalDownForm(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::uge || p == arith::CmpIPredicate::sge;
}

static bool isSemiOpenIntervalForm(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::ult || p == arith::CmpIPredicate::slt ||
         p == arith::CmpIPredicate::ne;
}

static bool validCountedLoopIntervalForm(arith::CmpIOp cmp,
                                         bool allowClosedInterval) {
  auto p = cmp.getPredicate();
  return isSemiOpenIntervalForm(p) ||
         (allowClosedInterval && isClosedIntervalForm(p));
}

// If the value, v, dominates the loop then it is invariant by definition.
// Block arguments that are, in fact, a threaded invariant value should have
// been converted to their dominating definition by the canonicalization
// pass, so a bare block argument that does not dominate the loop is not
// invariant (it is some value threaded through the CFG, e.g. a loop-carried
// argument).
//
// Otherwise, v may be defined by an op nested \em{inside} the loop that
// simply recomputes an invariant value on every iteration rather than
// having been hoisted out (e.g. `quake.veq_size` on a veq that is not
// modified by the loop). Such a value is still effectively invariant: if
// the op is free of memory effects and every one of its operands is
// (recursively) loop-invariant, then so is its result.
static bool isLoopInvariant(Value v, cudaq::cc::LoopOp loop,
                            const DominanceInfo &dom) {
  if (dom.dominates(v, loop.getOperation()))
    return true;
  auto *defOp = v.getDefiningOp();
  if (!defOp || !isMemoryEffectFree(defOp))
    return false;
  // A region-bearing op (e.g. a nested `cc.loop`) may be memory-effect-free
  // yet still capture values from its enclosing scope inside those regions —
  // values not reflected in `getOperands()` at all. Recursing only over the
  // explicit operand list would then wrongly call such an op "invariant"
  // even though a captured value only dominates the op's *original*
  // location. Callers that then hoist/clone this "invariant" value (see
  // materializeLoopInvariant) would silently produce a dominance violation.
  // Rather than reconstructing a whole region's capture set here, simply
  // refuse to treat any region-bearing op as invariant.
  if (defOp->getNumRegions() != 0)
    return false;
  return llvm::all_of(defOp->getOperands(), [&](Value operand) {
    return isLoopInvariant(operand, loop, dom);
  });
}

static bool isLoopInvariant(ArrayRef<Value> vs, cudaq::cc::LoopOp loop) {
  DominanceInfo dom(loop->getParentOfType<func::FuncOp>());
  return llvm::all_of(vs,
                      [&](Value v) { return isLoopInvariant(v, loop, dom); });
}

Value cudaq::opt::materializeLoopInvariant(RewriterBase &rewriter, Value v,
                                           cc::LoopOp loop) {
  DominanceInfo dom(loop->getParentOfType<func::FuncOp>());
  if (dom.dominates(v, loop.getOperation()))
    return v;
  auto *defOp = v.getDefiningOp();
  assert(defOp && isMemoryEffectFree(defOp) &&
         "v must be loop-invariant (see isLoopInvariant)");
  SmallVector<Value> newOperands;
  for (Value operand : defOp->getOperands())
    newOperands.push_back(materializeLoopInvariant(rewriter, operand, loop));
  Operation *clone = rewriter.clone(*defOp);
  clone->setOperands(newOperands);
  return clone->getResult(cast<OpResult>(v).getResultNumber());
}

/// Returns a pair `(true, stepValue)` if and only if the operation, \p op, is
/// an induction computation (integer add or subtract). Otherwise returns
/// `(false, null)`.
static std::pair<bool, Value> isInductionOn(unsigned offset, Operation *op,
                                            ArrayRef<BlockArgument> args) {
  if (auto addOp = dyn_cast_or_null<arith::AddIOp>(op)) {
    if (addOp.getLhs() == args[offset])
      return {true, addOp.getRhs()};
    if (addOp.getRhs() == args[offset])
      return {true, addOp.getLhs()};
  } else if (auto subOp = dyn_cast_or_null<arith::SubIOp>(op)) {
    if (subOp.getLhs() == args[offset])
      return {true, subOp.getRhs()};
  }
  return {false, Value{}};
}

// TODO: consider caching the results.
static BlockArgument getLinearExpr(Value expr,
                                   cudaq::opt::LoopComponents &result,
                                   cudaq::cc::LoopOp loop) {
  auto v = peelCastOps(expr);
  if (auto ba = dyn_cast<BlockArgument>(v)) {
    // Trivial expression: bare argument.
    return ba;
  }
  auto checkAndSet = [&](Value va, Value vb, Value &saved) -> BlockArgument {
    auto vl = peelCastOps(va);
    if (auto ba = dyn_cast<BlockArgument>(vl);
        ba && isLoopInvariant(vb, loop)) {
      saved = vb;
      return ba;
    }
    return {};
  };
  auto scaledIteration = [&](Value v) -> BlockArgument {
    if (auto mulOp = v.getDefiningOp<arith::MulIOp>()) {
      result.reciprocalScale = false;
      if (auto ba =
              checkAndSet(mulOp.getLhs(), mulOp.getRhs(), result.scaleValue))
        return ba;
      return checkAndSet(mulOp.getRhs(), mulOp.getLhs(), result.scaleValue);
    }
    if (auto divOp = v.getDefiningOp<arith::DivUIOp>()) {
      result.reciprocalScale = true;
      return checkAndSet(divOp.getLhs(), divOp.getRhs(), result.scaleValue);
    }
    if (auto divOp = v.getDefiningOp<arith::DivSIOp>()) {
      result.reciprocalScale = true;
      return checkAndSet(divOp.getLhs(), divOp.getRhs(), result.scaleValue);
    }
    return {};
  };
  if (auto addOp = expr.getDefiningOp<arith::AddIOp>()) {
    result.negatedAddend = false;
    result.minusOneMult = false;
    if (auto ba =
            checkAndSet(addOp.getLhs(), addOp.getRhs(), result.addendValue))
      return ba;
    if (auto ba = scaledIteration(addOp.getLhs());
        ba && isLoopInvariant({addOp.getRhs()}, loop)) {
      result.addendValue = addOp.getRhs();
      return ba;
    }
    if (auto ba =
            checkAndSet(addOp.getRhs(), addOp.getLhs(), result.addendValue))
      return ba;
    if (auto ba = scaledIteration(addOp.getRhs());
        ba && isLoopInvariant({addOp.getLhs()}, loop)) {
      result.addendValue = addOp.getLhs();
      return ba;
    }
    return {};
  }
  if (auto subOp = expr.getDefiningOp<arith::SubIOp>()) {
    if (auto ba =
            checkAndSet(subOp.getLhs(), subOp.getRhs(), result.addendValue)) {
      result.negatedAddend = true;
      return ba;
    }
    if (auto ba = scaledIteration(subOp.getLhs());
        ba && isLoopInvariant({subOp.getRhs()}, loop)) {
      result.addendValue = subOp.getRhs();
      result.negatedAddend = true;
      return ba;
    }
    if (auto ba =
            checkAndSet(subOp.getRhs(), subOp.getLhs(), result.addendValue)) {
      result.minusOneMult = true;
      return ba;
    }
    if (auto ba = scaledIteration(subOp.getRhs());
        ba && isLoopInvariant({subOp.getLhs()}, loop)) {
      result.addendValue = subOp.getLhs();
      result.minusOneMult = true;
      return ba;
    }
    return {};
  }
  return scaledIteration(expr);
}

static unsigned bitWidth(Value val) {
  return cast<IntegerType>(val.getType()).getWidth();
}

bool cudaq::opt::isSemiOpenPredicate(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::ult || p == arith::CmpIPredicate::slt ||
         p == arith::CmpIPredicate::ugt || p == arith::CmpIPredicate::sgt ||
         p == arith::CmpIPredicate::ne;
}

bool cudaq::opt::isUnsignedPredicate(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::ult || p == arith::CmpIPredicate::ule ||
         p == arith::CmpIPredicate::ugt || p == arith::CmpIPredicate::uge;
}

bool cudaq::opt::isSignedPredicate(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::slt || p == arith::CmpIPredicate::sle ||
         p == arith::CmpIPredicate::sgt || p == arith::CmpIPredicate::sge;
}

// We expect the loop control value to have the following form.
//
//   %final = cc.loop while ((%iter = %initial) -> (iN)) {
//     ...
//     %cond = arith.cmpi {<.<=,!=,>=,>}, %iter, %bound : iN
//     cc.condition %cond (%iter : iN)
//   } do {
//    ^bb1(%iter : iN):
//     ...
//     cc.continue %iter : iN
//   } step {
//    ^bb2(%iter : iN):
//     ...
//     %next = arith.{addi,subi} %iter, %step : iN
//     cc.continue %next : iN
//   }
//
// with the additional requirement that none of the `...` sections can modify
// the value of `%bound` or `%step`. Those values are invariant if there are
// no side-effects in the loop Op (no store or call operations) and these values
// do not depend on a block argument.
bool cudaq::opt::hasMonotonicControlInduction(cudaq::cc::LoopOp loop,
                                              cudaq::opt::LoopComponents *lcp) {
  if (loop.getInitialArgs().empty() || loop.getResults().empty())
    return false;
  if (auto c = getLoopComponents(loop)) {
    if (lcp)
      *lcp = *c;
    if (isLoopInvariant({c->compareValue, c->stepValue}, loop))
      return (bool)getLinearExpr(c->getCompareInduction(), *c, loop);
  }
  return false;
}

static bool allExitsAreContinue(Region &reg) {
  for (auto &block : reg)
    if (block.hasNoSuccessors() &&
        !isa<cudaq::cc::ContinueOp>(block.getTerminator()))
      return false;
  return true;
}

bool cudaq::opt::loopContainsBreak(cudaq::cc::LoopOp loopOp) {
  return !allExitsAreContinue(loopOp.getBodyRegion());
}

bool cudaq::opt::isaMonotonicLoop(Operation *op, bool allowEarlyExit,
                                  cudaq::opt::LoopComponents *lcp) {
  if (auto loopOp = dyn_cast_or_null<cc::LoopOp>(op)) {
    // Cannot be a `do while` loop. See cc-loop-peeling.
    if (loopOp.isPostConditional())
      return false;
    auto &reg = loopOp.getBodyRegion();
    return !reg.empty() && (allowEarlyExit || allExitsAreContinue(reg)) &&
           hasMonotonicControlInduction(loopOp, lcp);
  }
  return false;
}

bool cudaq::opt::isaInvariantLoop(const cudaq::opt::LoopComponents &c,
                                  bool allowClosedInterval) {
  if (isaConstantOf(c.initialValue, 0) && isaConstantOf(c.stepValue, 1) &&
      isa<arith::AddIOp>(c.stepOp)) {
    auto cmp = cast<arith::CmpIOp>(c.compareOp);
    return validCountedLoopIntervalForm(cmp, allowClosedInterval);
  }
  return false;
}

bool cudaq::opt::isaInvariantLoop(cudaq::cc::LoopOp loop,
                                  bool allowClosedInterval, bool allowEarlyExit,
                                  cudaq::opt::LoopComponents *lcp) {
  LoopComponents c;
  if (isaMonotonicLoop(loop.getOperation(), allowEarlyExit, &c)) {
    if (lcp)
      *lcp = c;
    return isaInvariantLoop(c, allowClosedInterval);
  }
  return false;
}

bool cudaq::opt::isaCountedLoop(cudaq::cc::LoopOp loop,
                                bool allowClosedInterval) {
  LoopComponents c;
  return isaInvariantLoop(loop, allowClosedInterval, /*allowEarlyExit=*/false,
                          &c) &&
         isaConstant(c.compareValue);
}

bool cudaq::opt::isaIndefiniteCountedLoop(cudaq::cc::LoopOp loop,
                                          bool allowClosedInterval) {
  LoopComponents c;
  return isaIndefiniteInvariantLoop(loop, allowClosedInterval, &c) &&
         isaConstant(c.compareValue);
}

mlir::Value cudaq::opt::LoopComponents::getCompareInduction() const {
  auto cmpOp = cast<arith::CmpIOp>(compareOp);
  return cmpOp.getLhs() == compareValue ? cmpOp.getRhs() : cmpOp.getLhs();
}

bool cudaq::opt::LoopComponents::stepIsAnAddOp() const {
  return isa<arith::AddIOp>(stepOp);
}

bool cudaq::opt::LoopComponents::shouldCommuteStepOp() const {
  if (auto addOp = dyn_cast_or_null<arith::AddIOp>(stepOp))
    if (induction.has_value())
      return addOp.getRhs() == stepRegion->front().getArgument(*induction);
  // Note: we don't allow induction on lhs of subtraction.
  return false;
}

bool cudaq::opt::LoopComponents::isClosedIntervalForm() const {
  auto p = cast<arith::CmpIOp>(compareOp).getPredicate();
  return ::isClosedIntervalForm(p) || ::isClosedIntervalDownForm(p);
}

bool cudaq::opt::LoopComponents::isLinearExpr() const {
  return addendValue || scaleValue;
}

std::int64_t cudaq::opt::LoopComponents::extendValue(unsigned width,
                                                     std::size_t val) const {
  const bool signExt =
      isSignedPredicate(cast<arith::CmpIOp>(compareOp).getPredicate());
  std::int64_t result = val;
  switch (width) {
  case 8:
    if (signExt) {
      std::int8_t v = val & 0xFF;
      result = v;
    } else {
      std::uint8_t v = val & 0xFF;
      result = v;
    }
    break;
  case 16:
    if (signExt) {
      std::int16_t v = val & 0xFFFF;
      result = v;
    } else {
      std::uint16_t v = val & 0xFFFF;
      result = v;
    }
    break;
  case 32:
    if (signExt) {
      std::int32_t v = val & 0xFFFFFFFF;
      result = v;
    } else {
      std::uint32_t v = val & 0xFFFFFFFF;
      result = v;
    }
    break;
  default:
    break;
  }
  return result;
}

bool cudaq::opt::LoopComponents::hasAlwaysTrueCondition() const {
  auto cmpValOpt = factory::maybeValueOfIntConstant(compareValue);
  if (!cmpValOpt)
    return false;
  auto width = bitWidth(compareValue);
  std::int64_t cmpVal = *cmpValOpt;
  auto pred = cast<arith::CmpIOp>(compareOp).getPredicate();
  switch (width) {
  case 8: {
    switch (pred) {
    case arith::CmpIPredicate::sge:
      return static_cast<std::int8_t>(cmpVal) ==
             std::numeric_limits<std::int8_t>::min();
    case arith::CmpIPredicate::sle:
      return static_cast<std::int8_t>(cmpVal) ==
             std::numeric_limits<std::int8_t>::max();
    case arith::CmpIPredicate::uge:
      return static_cast<std::uint8_t>(cmpVal) ==
             std::numeric_limits<std::uint8_t>::min();
    case arith::CmpIPredicate::ule:
      return static_cast<std::uint8_t>(cmpVal) ==
             std::numeric_limits<std::uint8_t>::max();
    default:
      break;
    }
  } break;
  case 16: {
    switch (pred) {
    case arith::CmpIPredicate::sge:
      return static_cast<std::int16_t>(cmpVal) ==
             std::numeric_limits<std::int16_t>::min();
    case arith::CmpIPredicate::sle:
      return static_cast<std::int16_t>(cmpVal) ==
             std::numeric_limits<std::int16_t>::max();
    case arith::CmpIPredicate::uge:
      return static_cast<std::uint16_t>(cmpVal) ==
             std::numeric_limits<std::uint16_t>::min();
    case arith::CmpIPredicate::ule:
      return static_cast<std::uint16_t>(cmpVal) ==
             std::numeric_limits<std::uint16_t>::max();
    default:
      break;
    }
  } break;
  case 32: {
    switch (pred) {
    case arith::CmpIPredicate::sge:
      return static_cast<std::int32_t>(cmpVal) ==
             std::numeric_limits<std::int32_t>::min();
    case arith::CmpIPredicate::sle:
      return static_cast<std::int32_t>(cmpVal) ==
             std::numeric_limits<std::int32_t>::max();
    case arith::CmpIPredicate::uge:
      return static_cast<std::uint32_t>(cmpVal) ==
             std::numeric_limits<std::uint32_t>::min();
    case arith::CmpIPredicate::ule:
      return static_cast<std::uint32_t>(cmpVal) ==
             std::numeric_limits<std::uint32_t>::max();
    default:
      break;
    }
  } break;
  case 64: {
    switch (pred) {
    case arith::CmpIPredicate::sge:
      return static_cast<std::int64_t>(cmpVal) ==
             std::numeric_limits<std::int64_t>::min();
    case arith::CmpIPredicate::sle:
      return static_cast<std::int64_t>(cmpVal) ==
             std::numeric_limits<std::int64_t>::max();
    case arith::CmpIPredicate::uge:
      return static_cast<std::uint64_t>(cmpVal) ==
             std::numeric_limits<std::uint64_t>::min();
    case arith::CmpIPredicate::ule:
      return static_cast<std::uint64_t>(cmpVal) ==
             std::numeric_limits<std::uint64_t>::max();
    default:
      break;
    }
  } break;
  default:
    break;
  }
  return false;
}

bool cudaq::opt::LoopComponents::hasAlwaysFalseCondition() const {
  auto cmpValOpt = factory::maybeValueOfIntConstant(compareValue);
  if (!cmpValOpt)
    return false;
  auto width = bitWidth(compareValue);
  std::int64_t cmpVal = *cmpValOpt;
  auto pred = cast<arith::CmpIOp>(compareOp).getPredicate();
  switch (width) {
  case 8: {
    switch (pred) {
    case arith::CmpIPredicate::slt:
      return static_cast<std::int8_t>(cmpVal) ==
             std::numeric_limits<std::int8_t>::min();
    case arith::CmpIPredicate::sgt:
      return static_cast<std::int8_t>(cmpVal) ==
             std::numeric_limits<std::int8_t>::max();
    case arith::CmpIPredicate::ult:
      return static_cast<std::uint8_t>(cmpVal) ==
             std::numeric_limits<std::uint8_t>::min();
    case arith::CmpIPredicate::ugt:
      return static_cast<std::uint8_t>(cmpVal) ==
             std::numeric_limits<std::uint8_t>::max();
    default:
      break;
    }
  } break;
  case 16: {
    switch (pred) {
    case arith::CmpIPredicate::slt:
      return static_cast<std::int16_t>(cmpVal) ==
             std::numeric_limits<std::int16_t>::min();
    case arith::CmpIPredicate::sgt:
      return static_cast<std::int16_t>(cmpVal) ==
             std::numeric_limits<std::int16_t>::max();
    case arith::CmpIPredicate::ult:
      return static_cast<std::uint16_t>(cmpVal) ==
             std::numeric_limits<std::uint16_t>::min();
    case arith::CmpIPredicate::ugt:
      return static_cast<std::uint16_t>(cmpVal) ==
             std::numeric_limits<std::uint16_t>::max();
    default:
      break;
    }
  } break;
  case 32: {
    switch (pred) {
    case arith::CmpIPredicate::slt:
      return static_cast<std::int32_t>(cmpVal) ==
             std::numeric_limits<std::int32_t>::min();
    case arith::CmpIPredicate::sgt:
      return static_cast<std::int32_t>(cmpVal) ==
             std::numeric_limits<std::int32_t>::max();
    case arith::CmpIPredicate::ult:
      return static_cast<std::uint32_t>(cmpVal) ==
             std::numeric_limits<std::uint32_t>::min();
    case arith::CmpIPredicate::ugt:
      return static_cast<std::uint32_t>(cmpVal) ==
             std::numeric_limits<std::uint32_t>::max();
    default:
      break;
    }
  } break;
  case 64: {
    switch (pred) {
    case arith::CmpIPredicate::slt:
      return static_cast<std::int64_t>(cmpVal) ==
             std::numeric_limits<std::int64_t>::min();
    case arith::CmpIPredicate::sgt:
      return static_cast<std::int64_t>(cmpVal) ==
             std::numeric_limits<std::int64_t>::max();
    case arith::CmpIPredicate::ult:
      return static_cast<std::uint64_t>(cmpVal) ==
             std::numeric_limits<std::uint64_t>::min();
    case arith::CmpIPredicate::ugt:
      return static_cast<std::uint64_t>(cmpVal) ==
             std::numeric_limits<std::uint64_t>::max();
    default:
      break;
    }
  } break;
  default:
    break;
  }
  return false;
}

std::optional<std::size_t>
cudaq::opt::LoopComponents::getIterationsConstant() const {
  auto initValOpt = factory::maybeValueOfIntConstant(initialValue);
  if (!initValOpt)
    return std::nullopt;
  std::int64_t initVal = extendValue(bitWidth(initialValue), *initValOpt);
  auto endValOpt = factory::maybeValueOfIntConstant(compareValue);
  if (!endValOpt)
    return std::nullopt;
  std::int64_t endVal = extendValue(bitWidth(compareValue), *endValOpt);
  auto stepValOpt = factory::maybeValueOfIntConstant(stepValue);
  if (!stepValOpt)
    return std::nullopt;
  std::int64_t stepVal = extendValue(bitWidth(stepValue), *stepValOpt);
  if (!stepIsAnAddOp())
    stepVal = -stepVal;
  if (isLinearExpr()) {
    if (addendValue) {
      auto addendOpt = factory::maybeValueOfIntConstant(addendValue);
      if (!addendOpt)
        return std::nullopt;
      std::int64_t addend = extendValue(bitWidth(addendValue), *addendOpt);
      if (negatedAddend)
        endVal += addend;
      else
        endVal -= addend;
    }
    if (minusOneMult) {
      initVal = -initVal;
      stepVal = -stepVal;
    }
    if (scaleValue) {
      auto scaleValOpt = factory::maybeValueOfIntConstant(scaleValue);
      if (!scaleValOpt)
        return std::nullopt;
      std::int64_t scaleVal = extendValue(bitWidth(scaleValue), *scaleValOpt);
      if (reciprocalScale) {
        endVal *= scaleVal;
      } else {
        endVal *= scaleVal;
        stepVal *= scaleVal;
      }
    }
  }
  if (!isClosedIntervalForm()) {
    if (stepVal < 0)
      endVal += 1;
    else
      endVal -= 1;
  }
  std::int64_t result = (endVal - initVal + stepVal) / stepVal;
  if (result < 0)
    result = 0;
  return {result};
}

template <typename T>
constexpr int computeArgsOffset() {
  if constexpr (std::is_same_v<T, cudaq::cc::ConditionOp>) {
    return 1;
  } else {
    return 0;
  }
}

std::optional<cudaq::opt::LoopComponents>
cudaq::opt::getLoopComponents(cudaq::cc::LoopOp loop) {
  LoopComponents result;
  auto &whileRegion = loop.getWhileRegion();
  auto &whileEntry = whileRegion.front();
  auto condOp = cast<cc::ConditionOp>(whileRegion.back().back());
  result.compareOp = condOp.getCondition().getDefiningOp();
  auto cmpOp = dyn_cast<arith::CmpIOp>(result.compareOp);
  if (!cmpOp)
    return {};

  auto argumentToCompare = [&](unsigned idx) -> bool {
    return (getLinearExpr(cmpOp.getLhs(), result, loop) ==
            whileEntry.getArgument(idx)) ||
           (getLinearExpr(cmpOp.getRhs(), result, loop) ==
            whileEntry.getArgument(idx));
  };
  auto scanRegionForStep =
      [&]<typename TERM, int argsOff = computeArgsOffset<TERM>()>(
          Region &reg) -> std::optional<unsigned> {
    // Pre-scan to make sure all terminators are ContinueOp.
    for (auto &block : reg)
      if (block.hasNoSuccessors())
        if (!isa<TERM>(block.back()))
          return {};

    for (auto &block : reg) {
      if (block.hasNoSuccessors()) {
        if (auto contOp = cast<TERM>(block.back())) {
          // Find an argument to the ContinueOp that is an integral induction
          // and updated by a step value.
          for (auto pr :
               llvm::enumerate(contOp.getOperands().drop_front(argsOff))) {
            if (auto *defOp = pr.value().getDefiningOp()) {
              if ((defOp->getBlock() == &block) &&
                  isa<arith::AddIOp, arith::SubIOp>(defOp)) {
                auto ps = isInductionOn(pr.index(), defOp,
                                        reg.front().getArguments());
                if (ps.first && argumentToCompare(pr.index())) {
                  // Set the step value and step op here.
                  result.stepValue = ps.second;
                  result.stepOp = defOp;
                  result.stepRegion = &reg;
                  return pr.index();
                }
              }
            }
          }
        }
      }
    }
    return {};
  };

  if (loop.hasStep()) {
    // Loop has a step region, so look for the step op.
    // as in: `for (i = 0; i < n; i++) ...`
    if (auto stepPosOpt = scanRegionForStep.template operator()<cc::ContinueOp>(
            loop.getStepRegion()))
      result.induction = *stepPosOpt;
  }
  if (!result.stepOp) {
    // If step has not been found, look in the body region.
    // as in: `for (i = 0; i < n;) { ... i++; }`
    if (auto stepPosOpt = scanRegionForStep.template operator()<cc::ContinueOp>(
            loop.getBodyRegion()))
      result.induction = *stepPosOpt;
  }
  if (!result.stepOp) {
    // If step has still not been found, look in the while region.
    // as in: `for (i = n; i-- > 0;) ...`
    if (auto stepPosOpt =
            scanRegionForStep.template operator()<cc::ConditionOp>(whileRegion))
      result.induction = *stepPosOpt;
  }
  if (!result.stepOp)
    return {};

  result.initialValue = loop.getInitialArgs()[*result.induction];

  // The comparison operation allows for the induction value to appear as part
  // of a loop-invariant linear expression on one side of the comparison. This
  // allows for invariant expressions on each side, such as, `4 * i + 1 < exp`.
  // This relaxation to invariant expressions requires some transformations to
  // normalize the comparison operation. Taking the example, this would
  // transform to `i < (exp - 1) / 4`.
  // TODO: A possible extension is to detect \em{conditionally iterated} loops
  // and open those up to further analysis and transformations such as loop
  // unrolling.
  if (getLinearExpr(cmpOp.getLhs(), result, loop) ==
      whileEntry.getArgument(*result.induction))
    result.compareValue = cmpOp.getRhs();
  else if (getLinearExpr(cmpOp.getRhs(), result, loop) ==
           whileEntry.getArgument(*result.induction))
    result.compareValue = cmpOp.getLhs();
  else
    return {};
  return result;
}

namespace {
/// Which region of a `cc.loop` to look in, and how to read its terminator's
/// carried operands: `cc.condition($cond, $results...)` for the while region
/// (the leading condition operand must be skipped), `cc.continue($operands...)`
/// for the body and step regions.
struct LoopRegionSite {
  Region *region;
  bool isWhile;
};
} // namespace

/// Returns the single no-successor (exit) block of \p reg, or null if there
/// isn't exactly one.
static Block *getSingleExitBlock(Region &reg) {
  Block *exit = nullptr;
  for (auto &block : reg) {
    if (block.hasNoSuccessors()) {
      if (exit)
        return nullptr;
      exit = &block;
    }
  }
  return exit;
}

/// Returns the value \p reg's terminator carries forward for loop-arg index
/// \p i (dropping the leading condition operand first, for the while
/// region), or a null Value if \p reg's exit terminator isn't the expected
/// kind or doesn't have that many operands.
static Value getCarriedValue(const LoopRegionSite &site, unsigned i) {
  Block *exit = getSingleExitBlock(*site.region);
  if (!exit)
    return {};
  if (site.isWhile) {
    auto cond = dyn_cast<cudaq::cc::ConditionOp>(exit->back());
    if (!cond || i >= cond.getResults().size())
      return {};
    return cond.getResults()[i];
  }
  auto cont = dyn_cast<cudaq::cc::ContinueOp>(exit->back());
  if (!cont || i >= cont.getOperands().size())
    return {};
  return cont.getOperands()[i];
}

SmallVector<cudaq::opt::SecondaryInduction>
cudaq::opt::getSecondaryInductions(cudaq::cc::LoopOp loop,
                                   const cudaq::opt::LoopComponents &primary) {
  SmallVector<SecondaryInduction> result;
  // Requires the primary to have been identified in a concrete region.
  if (!primary.induction.has_value())
    return result;
  unsigned primaryIdx = *primary.induction;

  // A secondary induction's own step need not live in the same region as
  // the primary's step: e.g. a `for (i = 0; i < n; i++) { ...; j += 1; }`
  // shaped loop has the primary `i` stepped in the step region while a
  // secondary `j` steps in the body region instead. Check every region the
  // loop has; whichever one actually modifies index `i` (if exactly one
  // does) supplies the step, and every other region must be a pure
  // passthrough (the carried value is literally that region's own entry
  // block argument) for the single closed-form step to be valid.
  SmallVector<LoopRegionSite> sites;
  sites.push_back({&loop.getWhileRegion(), /*isWhile=*/true});
  sites.push_back({&loop.getBodyRegion(), /*isWhile=*/false});
  if (loop.hasStep())
    sites.push_back({&loop.getStepRegion(), /*isWhile=*/false});

  unsigned numArgs = loop.getBodyRegion().front().getNumArguments();
  for (unsigned i = 0; i < numArgs; ++i) {
    if (i == primaryIdx)
      continue;

    Value stepVal;
    bool isAdd = false;
    bool isPrimaryAlias = false;
    unsigned steppingSitesFound = 0;
    bool malformed = false;

    for (auto &site : sites) {
      if (i >= site.region->front().getNumArguments() ||
          primaryIdx >= site.region->front().getNumArguments()) {
        malformed = true;
        break;
      }
      Value entryArg = site.region->front().getArgument(i);
      Value carried = getCarriedValue(site, i);
      if (!carried) {
        malformed = true;
        break;
      }
      if (carried == entryArg)
        continue; // Pure passthrough in this region — nothing to check.

      // The arg is reassigned each iteration to the primary induction's own
      // per-iteration value, e.g. `cc.continue %i, %i, ...`. Its closed form
      // then rides on the primary's own step rather than an independent
      // accumulation; see the `aliasesPrimary` field.
      if (carried == site.region->front().getArgument(primaryIdx)) {
        Value candStep = primary.stepValue;
        bool candIsAdd = primary.stepIsAnAddOp();
        if (!candStep || !isLoopInvariant({candStep}, loop)) {
          malformed = true;
          break;
        }
        ++steppingSitesFound;
        stepVal = candStep;
        isAdd = candIsAdd;
        isPrimaryAlias = true;
        continue;
      }

      Block *exit = getSingleExitBlock(*site.region);
      auto *defOp = carried.getDefiningOp();
      if (!defOp || defOp->getBlock() != exit) {
        malformed = true;
        break;
      }

      Value candStep;
      bool candIsAdd = false;
      if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
        if (addOp.getLhs() == entryArg) {
          candStep = addOp.getRhs();
          candIsAdd = true;
        } else if (addOp.getRhs() == entryArg) {
          candStep = addOp.getLhs();
          candIsAdd = true;
        }
      } else if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
        if (subOp.getLhs() == entryArg) {
          candStep = subOp.getRhs();
          candIsAdd = false;
        }
      }
      if (!candStep || !isLoopInvariant({candStep}, loop)) {
        malformed = true;
        break;
      }

      ++steppingSitesFound;
      stepVal = candStep;
      isAdd = candIsAdd;
      isPrimaryAlias = false;
    }

    // Fuse only when index `i` is stepped in exactly one region and left
    // untouched everywhere else — anything else (stepped in more than one
    // region, or a shape getCarriedValue/defOp can't make sense of) isn't a
    // simple induction we can safely turn into a closed form, so leave it
    // loop-carried rather than risk fusing it incorrectly.
    if (malformed || steppingSitesFound != 1)
      continue;

    SecondaryInduction ind;
    ind.argIndex = i;
    ind.initialValue = loop.getInitialArgs()[i];
    ind.stepValue = stepVal;
    ind.stepIsAdd = isAdd;
    ind.aliasesPrimary = isPrimaryAlias;
    result.push_back(ind);
  }
  return result;
}
