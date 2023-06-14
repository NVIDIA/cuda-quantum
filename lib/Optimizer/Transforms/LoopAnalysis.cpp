/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"

using namespace mlir;

/// \file
/// Some working definitions:
///
/// A \em counted loop, we mean a loop that counts from `0` up to `n-1` stepping
/// by 1. Such a loop is \em normalized (starts at 0), \em monotonically
/// increasing (slope is a constant 1), executes exactly `n` times, and `n` is a
/// compile-time constant. A counted loop is said to have static control flow.
///
/// An \em invariant loop is a counted loop but `n` need not be a compile-time
/// constant. An invariant loop cannot be fully unrolled until runtime. In
/// quantum circuit speak, one does not know the full size of the circuit.
///
/// A \em monotonic loop is a loop that counts from `i` up to (down to) `j`
/// stepping by positive (negative) integral values; mathematically, it is a
/// strictly monotonic sequence. If the step is a compile-time constant, `k`,
/// then a closed iterval monotonic loop must execute exactly `floor((j - i + k)
/// / k)` iterations. By normalizing a monotonic loop and constant folding and
/// propagation, we may be able to convert it to static control flow.
///
/// For completeness, a \em{conditionally iterated} loop is a monotonic loop
/// that has a second auxilliary condition to determine if a given loop
/// iteration is executed or not. For example, the condition might be used in
/// iteration `m` to disable all subsequent iterations. (Much like a `break`
/// statement.) These loops might be unrolled but only if the loop can be
/// normalized into static control flow and the auxillary condition can be
/// computed as a constant. It is likely these loops cannot be converted to
/// static control flow and would thus need to be expanded at runtime.

static bool isaConstant(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    return isa<IntegerAttr>(c.getValue());
  return false;
}

static bool isaConstantOf(Value v, std::int64_t hasVal) {
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt() == hasVal;
  return false;
}

static bool isClosedIntervalForm(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::ule || p == arith::CmpIPredicate::sle;
}

static bool isSemiOpenIntervalForm(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::ult || p == arith::CmpIPredicate::slt ||
         p == arith::CmpIPredicate::ne;
}

static bool validCountedLoopIntervalForm(arith::CmpIOp cmp,
                                         bool allowClosedInterval) {
  auto p = cmp.getPredicate();
  return isSemiOpenIntervalForm(p) ||
         allowClosedInterval && isClosedIntervalForm(p);
}

namespace cudaq {

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
// FIXME: assumes only the LCV is passed as a Value.
bool opt::hasMonotonicControlInduction(cc::LoopOp loop) {
  if (loop.getInitialArgs().empty() || loop.getResults().empty())
    return false;
  auto &whileBlock = loop.getWhileRegion().back();
  auto condition = dyn_cast<cc::ConditionOp>(whileBlock.back());
  if (!condition || whileBlock.getArguments()[0] != condition.getResults()[0])
    return false;
  auto *cmpOp = condition.getCondition().getDefiningOp();
  if (std::find(cmpOp->getOperands().begin(), cmpOp->getOperands().end(),
                whileBlock.getArguments()[0]) == cmpOp->getOperands().end())
    return false;
  auto &bodyBlock = loop.getBodyRegion().back();
  auto bodyTermOp = dyn_cast<cc::ContinueOp>(bodyBlock.back());
  if (!bodyTermOp || (bodyBlock.getArguments()[0] != bodyTermOp.getOperand(0)))
    return false;
  auto &stepBlock = loop.getStepRegion().back();
  auto backedgeOp = dyn_cast<cc::ContinueOp>(stepBlock.back());
  if (!backedgeOp)
    return false;
  auto *mutateOp = backedgeOp.getOperand(0).getDefiningOp();
  if (!isa<arith::AddIOp, arith::SubIOp>(mutateOp) ||
      std::find(mutateOp->getOperands().begin(), mutateOp->getOperands().end(),
                stepBlock.getArguments()[0]) == mutateOp->getOperands().end())
    return false;
  // FIXME: should verify %bound, %step are loop invariant.
  return true;
}

bool opt::isaMonotonicLoop(Operation *op) {
  if (auto loopOp = dyn_cast_or_null<cc::LoopOp>(op)) {
    // Cannot be a `while` or `do while` loop.
    if (loopOp.isPostConditional() || !loopOp.hasStep())
      return false;
    auto &reg = loopOp.getBodyRegion();
    // This is a `for` loop and must have a body with a continue terminator.
    // Currently, only a single basic block is allowed to keep things simple.
    // This is in keeping with our definition of structured control flow.
    return !reg.empty() && reg.hasOneBlock() &&
           isa<cc::ContinueOp>(reg.front().getTerminator()) &&
           hasMonotonicControlInduction(loopOp);
  }
  return false;
}

bool opt::isaCountedLoop(cc::LoopOp loop, bool allowClosedInterval) {
  if (isaMonotonicLoop(loop.getOperation())) {
    if (auto components = getLoopComponents(loop)) {
      auto &c = *components;
      if (isaConstantOf(c.initialValue, 0) && isaConstant(c.compareValue) &&
          isaConstantOf(c.stepValue, 1) && isa<arith::AddIOp>(c.stepOp)) {
        auto cmp = cast<arith::CmpIOp>(c.compareOp);
        return validCountedLoopIntervalForm(cmp, allowClosedInterval);
      }
    }
  }
  return false;
}

bool opt::LoopComponents::stepIsAnAddOp() { return isa<arith::AddIOp>(stepOp); }

bool opt::LoopComponents::shouldCommuteStepOp() {
  if (auto addOp = dyn_cast_or_null<arith::AddIOp>(stepOp))
    return addOp.getRhs() == stepRegion->front().getArgument(induction);
  // Note: we don't allow induction on lhs of subtraction.
  return false;
}

bool opt::LoopComponents::isClosedIntervalForm() {
  auto cmp = cast<arith::CmpIOp>(compareOp);
  return ::isClosedIntervalForm(cmp.getPredicate());
}

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

std::optional<opt::LoopComponents> opt::getLoopComponents(cc::LoopOp loop) {
  opt::LoopComponents result;
  auto &whileRegion = loop.getWhileRegion();
  auto condOp = cast<cc::ConditionOp>(whileRegion.back().back());
  result.compareOp = condOp.getCondition().getDefiningOp();
  auto cmpOp = cast<arith::CmpIOp>(result.compareOp);

  auto scanRegionForStep = [&](Region &reg) -> std::optional<unsigned> {
    std::optional<unsigned> res;
    for (auto &block : reg) {
      if (block.hasNoSuccessors()) {
        if (auto contOp = dyn_cast<cc::ContinueOp>(block.back())) {
          // Find an argument to the ContinueOp that is an integral induction
          // and updated by a step value.
          for (auto pr : llvm::enumerate(contOp.getOperands())) {
            if (auto *defOp = pr.value().getDefiningOp()) {
              if ((defOp->getBlock() == &block) &&
                  isa<arith::AddIOp, arith::SubIOp>(defOp)) {
                auto ps = isInductionOn(pr.index(), defOp,
                                        reg.front().getArguments());
                if (ps.first) {
                  // Set the step value and step op here.
                  result.stepValue = ps.second;
                  result.stepOp = defOp;
                  result.stepRegion = &reg;
                  if (!res)
                    res = pr.index();
                  else
                    return {}; // LoopOp has unexpected induction(s).
                }
              }
            }
          }
        } else {
          return {}; // LoopOp is malformed.
        }
      }
    }
    return {};
  };

  if (loop.hasStep()) {
    // Loop has a step region, so look for the step op.
    if (auto stepPosOpt = scanRegionForStep(loop.getStepRegion()))
      result.induction = *stepPosOpt;
  }
  // If step has not been found, look in the body region.
  if (!result.stepOp)
    if (auto stepPosOpt = scanRegionForStep(loop.getBodyRegion()))
      result.induction = *stepPosOpt;
  if (!result.stepOp)
    return {};

  result.initialValue = loop.getInitialArgs()[result.induction];
  if (cmpOp.getLhs() ==
      loop.getWhileRegion().front().getArgument(result.induction))
    result.compareValue = cmpOp.getRhs();
  else if (cmpOp.getRhs() ==
           loop.getWhileRegion().front().getArgument(result.induction))
    result.compareValue = cmpOp.getLhs();
  else
    return {};
  return result;
}

} // namespace cudaq
