/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "PhaseUtilities.h"
#include "QuakeOperatorCreator.h"
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Transforms/CommutationAwareRewrite.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKESIMPLIFY
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "quake-simplify"

using namespace mlir;

static cudaq::cc::ScopeOp getEnclosingAtomicQuantumRegion(Operation *op) {
  for (auto parentScope = op->getParentOfType<cudaq::cc::ScopeOp>();
       parentScope;
       parentScope = parentScope->getParentOfType<cudaq::cc::ScopeOp>())
    if (parentScope.getAtomicQuantumRegionAttr())
      return parentScope;
  return {};
}

// Enforce the atomic-region optimization contract: a pattern may combine
// two operations only when they have the same nearest enclosing
// `atomic_quantum_region` scope and every region boundary between them is an
// ordinary single-block `cc.scope`.
static bool shareOptimizationRegion(Operation *later, Operation *earlier) {
  if (getEnclosingAtomicQuantumRegion(later) !=
      getEnclosingAtomicQuantumRegion(earlier))
    return false;

  // Defining operations may be visible from nested regions. Only ordinary,
  // single-block scopes are transparent to these local rewrites.
  Block *nested = later->getBlock();
  Block *outer = earlier->getBlock();
  while (nested != outer) {
    if (!nested)
      return false;
    auto scope = dyn_cast_or_null<cudaq::cc::ScopeOp>(nested->getParentOp());
    if (!scope || scope.getAtomicQuantumRegionAttr() ||
        !scope.getInitRegion().hasOneBlock())
      return false;
    nested = scope->getBlock();
  }
  return true;
}

// Compare how two operations control, not which qubits they control: the number
// of controls and each one's polarity. Which qubits fill those positions is the
// matcher's question. An absent polarity attribute means every control is
// positive.
static bool
haveSameControlArityAndPolarity(cudaq::quake::OperatorInterface lhs,
                                cudaq::quake::OperatorInterface rhs) {
  auto lhsControls = lhs.getControls();
  auto rhsControls = rhs.getControls();
  if (lhsControls.size() != rhsControls.size())
    return false;

  auto lhsPolarities = lhs.getNegatedControls();
  auto rhsPolarities = rhs.getNegatedControls();
  for (std::size_t i = 0, e = lhsControls.size(); i != e; ++i) {
    bool lhsNegated = lhsPolarities && (*lhsPolarities)[i];
    bool rhsNegated = rhsPolarities && (*rhsPolarities)[i];
    if (lhsNegated != rhsNegated)
      return false;
  }
  return true;
}

// MLIR canonicalization can temporarily duplicate wire uses across blocks, and
// Quake's verifier accepts that degraded form for later linearity repair.
// Require every producer result to have exactly one use before rewriting or
// erasing it.
static bool shouldSkipRewrite(Operation *later, Operation *earlier) {
  return !shareOptimizationRegion(later, earlier) ||
         !llvm::all_of(earlier->getResults(),
                       [](Value result) { return result.hasOneUse(); });
}

#include "RewriteRotationsToCliffordT.inc"

template <typename QOP>
static QOP
getSameActionEndpoint(QOP later, Operation *candidate,
                      cudaq::opt::CommutationAwareRewriteMatcher &matcher) {
  auto earlier = dyn_cast<QOP>(candidate);
  if (!earlier || !haveSameControlArityAndPolarity(later, earlier) ||
      !matcher.have_same_ordered_quantum_operands(later, earlier))
    return {};
  return earlier;
}

// Preserve adjacent inverse pairs across one ordinary single-block `cc.scope`,
// where the block-local matcher must stop. Empty physical intervals on both
// sides of the boundary prevent hidden effects through aliases. Exact
// result-to-operand threading proves the ordered roles, while the matcher
// proves that every predecessor role has a distinct logical identity. Atomic
// and unsupported boundaries remain opaque.
template <typename QOP>
static QOP
getCrossScopePredecessor(QOP later,
                         cudaq::opt::CommutationAwareRewriteMatcher &matcher) {
  auto laterFlow = cudaq::quake::detail::getScalarWireFlow(later);
  if (!laterFlow || laterFlow->inputs.empty())
    return {};

  auto scope = dyn_cast_or_null<cudaq::cc::ScopeOp>(later->getParentOp());
  auto predecessor = laterFlow->inputs.front().template getDefiningOp<QOP>();
  if (!scope || scope.getAtomicQuantumRegionAttr() ||
      !scope.getInitRegion().hasOneBlock() ||
      later->getBlock() != &scope.getInitRegion().front() || !predecessor ||
      predecessor->getBlock() != scope->getBlock() ||
      predecessor->getNextNode() != scope.getOperation() ||
      later->getPrevNode() || shouldSkipRewrite(later, predecessor) ||
      !haveSameControlArityAndPolarity(predecessor, later) ||
      !matcher.has_distinct_quantum_operands(predecessor))
    return {};

  return predecessor;
}

template <typename QOP>
static bool hasOrderedResultThreading(QOP predecessor, QOP later) {
  auto laterFlow = cudaq::quake::detail::getScalarWireFlow(later);
  auto predecessorFlow = cudaq::quake::detail::getScalarWireFlow(predecessor);
  return laterFlow && predecessorFlow &&
         llvm::equal(predecessorFlow->results, laterFlow->inputs);
}

// Reference-form operators have no results. Value-form operators forward each
// scalar wire control and target to the result that carries the same qubit.
// Reusable controls are not consumed and therefore have no forwarded result.
static void eraseOrForward(cudaq::quake::OperatorInterface operation,
                           PatternRewriter &rewriter) {
  if (operation->getNumResults() == 0) {
    rewriter.eraseOp(operation);
    return;
  }

  llvm::SmallVector<Value> wires;
  for (Value control : operation.getControls())
    if (isa<cudaq::quake::WireType>(control.getType()))
      wires.push_back(control);
  for (Value target : operation.getTargets())
    if (isa<cudaq::quake::WireType>(target.getType()))
      wires.push_back(target);
  assert(wires.size() == operation->getNumResults() &&
         "operator results must correspond to scalar wire operands");
  rewriter.replaceOp(operation, wires);
}

// Splice both endpoints out of their wires. Replacing the later operation first
// preserves the greedy anchor, and forwarding each operation's own operands
// leaves the crossed commuting operations in place.
template <typename QOP>
static void cancelPair(QOP earlier, QOP later, PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "eliminated: " << earlier << '\n'
                          << later << '\n');
  eraseOrForward(later, rewriter);
  eraseOrForward(earlier, rewriter);
}

template <typename QOP, typename IsEndpoint>
static LogicalResult
cancelTransparentPair(QOP later,
                      cudaq::opt::CommutationAwareRewriteMatcher &matcher,
                      PatternRewriter &rewriter, IsEndpoint isEndpoint) {
  Operation *earlier = matcher.find_nearest(later, isEndpoint);
  if (!earlier)
    return failure();

  cancelPair(cast<QOP>(earlier), later, rewriter);
  return success();
}

enum class InversePairKind { SelfInverse, OppositeAdjoints };

template <typename QOP, InversePairKind Kind = InversePairKind::SelfInverse>
class InverseElimination : public OpRewritePattern<QOP> {
public:
  InverseElimination(MLIRContext *context,
                     cudaq::opt::CommutationAwareRewriteMatcher &matcher,
                     Pass::Statistic &stat)
      : OpRewritePattern<QOP>(context), matcher(matcher), stat(stat) {}

  LogicalResult matchAndRewrite(QOP later,
                                PatternRewriter &rewriter) const override {
    auto result = cancelTransparentPair(
        later, matcher, rewriter, [&](Operation *candidate) {
          auto earlier = getSameActionEndpoint(later, candidate, matcher);
          return earlier && (Kind == InversePairKind::SelfInverse ||
                             later.isAdj() != earlier.isAdj());
        });
    if (succeeded(result)) {
      ++stat;
      return success();
    }

    auto predecessor = getCrossScopePredecessor(later, matcher);
    if (!predecessor || !hasOrderedResultThreading(predecessor, later) ||
        (Kind == InversePairKind::OppositeAdjoints &&
         later.isAdj() == predecessor.isAdj()))
      return failure();

    cancelPair(predecessor, later, rewriter);
    ++stat;
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  Pass::Statistic &stat;
};

// Swap is symmetric in its two targets, so a later operation that names them in
// the opposite order still inverts the earlier operation. Match this case
// positionally because the ordered-identity query cannot express it. The later
// wire operands must be the earlier results with the final two target positions
// transposed. Linear use means this is necessarily adjacent.
static bool hasTransposedTargetsOnEarlierWires(cudaq::quake::SwapOp earlier,
                                               cudaq::quake::SwapOp later) {
  auto earlierFlow = cudaq::quake::detail::getScalarWireFlow(earlier);
  auto laterFlow = cudaq::quake::detail::getScalarWireFlow(later);
  if (!earlierFlow || !laterFlow)
    return false;
  const auto &laterWires = laterFlow->inputs;
  const auto &earlierWires = earlierFlow->results;
  if (laterWires.size() != earlierWires.size() || earlierWires.size() < 2)
    return false;

  for (std::size_t i = 0, e = earlierWires.size() - 2; i != e; ++i)
    if (laterWires[i] != earlierWires[i])
      return false;
  return laterWires[laterWires.size() - 2] == earlierWires.back() &&
         laterWires.back() == earlierWires[earlierWires.size() - 2];
}

template <>
class InverseElimination<cudaq::quake::SwapOp>
    : public OpRewritePattern<cudaq::quake::SwapOp> {
public:
  InverseElimination(MLIRContext *context,
                     cudaq::opt::CommutationAwareRewriteMatcher &matcher,
                     Pass::Statistic &stat)
      : OpRewritePattern<cudaq::quake::SwapOp>(context), matcher(matcher),
        stat(stat) {}

  LogicalResult matchAndRewrite(cudaq::quake::SwapOp later,
                                PatternRewriter &rewriter) const override {
    auto result = cancelTransparentPair(
        later, matcher, rewriter, [&](Operation *candidate) {
          auto earlier = dyn_cast<cudaq::quake::SwapOp>(candidate);
          return earlier && haveSameControlArityAndPolarity(later, earlier) &&
                 (matcher.have_same_ordered_quantum_operands(later, earlier) ||
                  hasTransposedTargetsOnEarlierWires(earlier, later));
        });
    if (succeeded(result)) {
      ++stat;
      return success();
    }

    auto predecessor = getCrossScopePredecessor(later, matcher);
    if (!predecessor ||
        (!hasOrderedResultThreading(predecessor, later) &&
         !hasTransposedTargetsOnEarlierWires(predecessor, later)))
      return failure();

    cancelPair(predecessor, later, rewriter);
    ++stat;
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  Pass::Statistic &stat;
};

// The angle, in radians, after which QOP is exactly the identity operator. The
// axis rotations are spinors: they pick up an overall `-1` at `2*pi` and return
// to the identity only after `4*pi`. `r1` is `diag(1, exp(i*theta))` and has
// period `2*pi` outright.
template <typename QOP>
constexpr double exactIdentityPeriod() {
  if constexpr (std::is_same_v<QOP, cudaq::quake::R1Op>)
    return 2.0 * M_PI;
  return 4.0 * M_PI;
}

// Preserve `quake.ry(12 * pi)` as an identity marker until Quake provides an
// explicit identity operation.
template <typename QOP>
static bool isReservedRyMarker(double theta) {
  return std::is_same_v<QOP, cudaq::quake::RyOp> && theta == 12.0 * M_PI;
}

template <typename QOP>
static bool isIdentityRotation(QOP qop, double threshold) {
  Attribute attr;
  if (!matchPattern(qop.getParameters().front(), m_Constant(&attr)))
    return false;

  // Normalize exactly representable constants to double; reject lossy
  // conversions.
  APFloat angle = cast<FloatAttr>(attr).getValue();
  bool lostPrecision = false;
  if (angle.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                    &lostPrecision) != APFloat::opOK ||
      lostPrecision)
    return false;
  double theta = angle.convertToDouble();

  if (isReservedRyMarker<QOP>(theta))
    return false;

  // Never the shorter period: the `-1` an axis rotation picks up at `2*pi` is
  // observable if the rotation runs under a control, and this op may yet be
  // given one by control synthesis of the function it is in.
  double residual = std::remainder(theta, exactIdentityPeriod<QOP>());

  // A positive threshold permits approximate identity elimination.
  return std::abs(residual) <= threshold;
}

static bool haveExactValue(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (lhs.getType() != rhs.getType())
    return false;

  Attribute lhsConstant;
  Attribute rhsConstant;
  return matchPattern(lhs, m_Constant(&lhsConstant)) &&
         matchPattern(rhs, m_Constant(&rhsConstant)) &&
         lhsConstant == rhsConstant;
}

static Value createCombinedRotationAngle(PatternRewriter &rewriter,
                                         Location location, Value laterAngle,
                                         bool laterIsAdjoint,
                                         Value earlierAngle,
                                         bool earlierIsAdjoint) {
  Type angleType = laterAngle.getType();
  if (laterIsAdjoint)
    laterAngle =
        arith::NegFOp::create(rewriter, location, angleType, laterAngle);
  if (earlierIsAdjoint)
    earlierAngle =
        arith::NegFOp::create(rewriter, location, angleType, earlierAngle);
  // Insertion at the later operation keeps both parameters dominant. Retain
  // the established floating-point association as later angle plus earlier
  // angle.
  return arith::AddFOp::create(rewriter, location, angleType, laterAngle,
                               earlierAngle);
}

// `phased_rx` folds only rotations with the same axis. Combine signed theta
// and preserve phi.
class PhasedRxCombine : public OpRewritePattern<cudaq::quake::PhasedRxOp> {
public:
  PhasedRxCombine(MLIRContext *context,
                  cudaq::opt::CommutationAwareRewriteMatcher &matcher,
                  double threshold, Pass::Statistic &zeroStat,
                  Pass::Statistic &combineStat)
      : OpRewritePattern<cudaq::quake::PhasedRxOp>(context), matcher(matcher),
        threshold(threshold), zeroStat(zeroStat), combineStat(combineStat) {}

  LogicalResult matchAndRewrite(cudaq::quake::PhasedRxOp later,
                                PatternRewriter &rewriter) const override {
    auto laterParameters = later.getParameters();
    if (laterParameters.size() != 2)
      return failure();

    if (isIdentityRotation(later, threshold)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "zero rotation eliminated [" << later << "]\n");
      eraseOrForward(later, rewriter);
      ++zeroStat;
      return success();
    }

    // Stop at the first structurally matching action. Checking phi afterward
    // keeps a different axis as a barrier instead of searching past it.
    Operation *match = matcher.find_nearest(later, [&](Operation *candidate) {
      return static_cast<bool>(
          getSameActionEndpoint(later, candidate, matcher));
    });
    if (!match)
      return failure();

    auto earlier = cast<cudaq::quake::PhasedRxOp>(match);
    auto earlierParameters = earlier.getParameters();
    if (earlierParameters.size() != 2 ||
        !haveExactValue(laterParameters[1], earlierParameters[1]) ||
        laterParameters.front().getType() !=
            earlierParameters.front().getType())
      return failure();

    Value laterTheta = laterParameters.front();
    Value earlierTheta = earlierParameters.front();
    if (later.isAdj() != earlier.isAdj() &&
        haveExactValue(laterTheta, earlierTheta)) {
      cancelPair(earlier, later, rewriter);
      ++combineStat;
      return success();
    }

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(later);
      Value combinedTheta = createCombinedRotationAngle(
          rewriter, later.getLoc(), laterTheta, later.isAdj(), earlierTheta,
          earlier.isAdj());

      LLVM_DEBUG(llvm::dbgs() << "combined: " << earlier << '\n'
                              << later << '\n');
      [[maybe_unused]] auto combined =
          rewriter.replaceOpWithNewOp<cudaq::quake::PhasedRxOp>(
              later, later.getResultTypes(), UnitAttr{},
              ValueRange{combinedTheta, laterParameters[1]},
              later.getControls(), later.getTargets(),
              later.getNegatedQubitControlsAttr());
      LLVM_DEBUG(llvm::dbgs() << "into: " << combined << '\n');
    }
    eraseOrForward(earlier, rewriter);
    ++combineStat;
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  double threshold;
  Pass::Statistic &zeroStat;
  Pass::Statistic &combineStat;
};

template <typename QOP>
class RotationCombine : public OpRewritePattern<QOP> {
public:
  RotationCombine(MLIRContext *context,
                  cudaq::opt::CommutationAwareRewriteMatcher &matcher,
                  double threshold, Pass::Statistic &zeroStat,
                  Pass::Statistic &combineStat)
      : OpRewritePattern<QOP>(context), matcher(matcher), threshold(threshold),
        zeroStat(zeroStat), combineStat(combineStat) {}

  LogicalResult matchAndRewrite(QOP later,
                                PatternRewriter &rewriter) const override {
    auto parameters = later.getParameters();
    if (parameters.size() != 1)
      return failure();

    if (isIdentityRotation(later, threshold)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "zero rotation eliminated [" << later << "]\n");
      eraseOrForward(later, rewriter);
      ++zeroStat;
      return success();
    }

    Value laterAngle = parameters.front();
    Operation *match = matcher.find_nearest(later, [&](Operation *candidate) {
      auto earlier = getSameActionEndpoint(later, candidate, matcher);
      if (!earlier)
        return false;
      auto earlierParameters = earlier.getParameters();
      return earlierParameters.size() == 1 &&
             earlierParameters.front().getType() == laterAngle.getType();
    });
    if (!match)
      return failure();

    auto earlier = cast<QOP>(match);
    Value earlierAngle = earlier.getParameters().front();
    if (later.isAdj() != earlier.isAdj() &&
        haveExactValue(laterAngle, earlierAngle)) {
      cancelPair(earlier, later, rewriter);
      ++combineStat;
      return success();
    }

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(later);
      Value combinedAngle = createCombinedRotationAngle(
          rewriter, later.getLoc(), laterAngle, later.isAdj(), earlierAngle,
          earlier.isAdj());

      LLVM_DEBUG(llvm::dbgs() << "combined: " << earlier << '\n'
                              << later << '\n');
      [[maybe_unused]] auto combined = rewriter.replaceOpWithNewOp<QOP>(
          later, later.getResultTypes(), UnitAttr{}, ValueRange{combinedAngle},
          later.getControls(), later.getTargets(),
          later.getNegatedQubitControlsAttr());
      LLVM_DEBUG(llvm::dbgs() << "into: " << combined << '\n');
    }
    eraseOrForward(earlier, rewriter);
    ++combineStat;
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  double threshold;
  Pass::Statistic &zeroStat;
  Pass::Statistic &combineStat;
};

// Z = SS = S<adj>S<adj>; S = TT; S<adj> = T<adj>T<adj>.
template <typename SourceOp, typename FoldedOp>
class DiscretePhaseFold : public OpRewritePattern<SourceOp> {
public:
  DiscretePhaseFold(MLIRContext *context,
                    cudaq::opt::CommutationAwareRewriteMatcher &matcher,
                    Pass::Statistic &stat)
      : OpRewritePattern<SourceOp>(context), matcher(matcher), stat(stat) {}

  LogicalResult matchAndRewrite(SourceOp later,
                                PatternRewriter &rewriter) const override {
    Operation *match = matcher.find_nearest(later, [&](Operation *candidate) {
      return static_cast<bool>(
          getSameActionEndpoint(later, candidate, matcher));
    });
    if (!match)
      return failure();

    auto earlier = cast<SourceOp>(match);
    // Let the inverse-elimination pattern own the opposite-adjoint pair.
    if (later.isAdj() != earlier.isAdj())
      return failure();

    UnitAttr foldedAdjoint;
    if constexpr (std::is_same_v<FoldedOp, cudaq::quake::SOp>)
      foldedAdjoint = later.getIsAdjAttr();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(later);
      rewriter.replaceOpWithNewOp<FoldedOp>(
          later, later.getResultTypes(), foldedAdjoint, ValueRange{},
          later.getControls(), later.getTargets(),
          later.getNegatedQubitControlsAttr());
    }
    eraseOrForward(earlier, rewriter);
    ++stat;
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  Pass::Statistic &stat;
};

// X S Y = -S; X S<adj> Y = S<adj>.
class ReduceYSX : public OpRewritePattern<cudaq::quake::XOp> {
public:
  ReduceYSX(MLIRContext *context, Pass::Statistic &stat)
      : OpRewritePattern(context), stat(stat) {}

  LogicalResult matchAndRewrite(cudaq::quake::XOp qop,
                                PatternRewriter &rewriter) const override {
    auto targets = qop.getTargets();
    if (targets.size() != 1 ||
        !isa<cudaq::quake::WireType>(targets.front().getType())) {
      LLVM_DEBUG(llvm::dbgs() << "operation must have 1 target\n");
      return failure();
    }
    Value trgt = targets.front();

    auto prev0 = targets.front().template getDefiningOp<cudaq::quake::SOp>();
    if (!prev0 || prev0.getTargets().size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must be S\n");
      return failure();
    }
    auto prev =
        prev0.getTargets().front().template getDefiningOp<cudaq::quake::YOp>();
    if (!prev || prev.getTargets().size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "previous previous operation must be Y\n");
      return failure();
    }

    // Scalar def-use cannot exclude effects through another SSA or reference
    // alias. Accept only empty Y/S/X intervals, conservatively missing safe
    // nonempty intervals.
    if (prev->getNextNode() != prev0.getOperation() ||
        prev0->getNextNode() != qop.getOperation())
      return failure();

    if (shouldSkipRewrite(qop, prev0) || shouldSkipRewrite(qop, prev))
      return failure();

    // Check target is properly threaded.
    auto prev0Trgs = prev0.getTargets();
    auto prevTrgs = prev.getTargets();
    if (prev0Trgs.size() != 1 || prevTrgs.size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must have 1 target\n");
      return failure();
    }
    Value prev0Trgt = prev0Trgs[0];
    Value prevTrgt = prevTrgs[0];
    auto prev0Wires = prev0.getWires();
    auto prevWires = prev.getWires();
    if (!isa<cudaq::quake::WireType>(trgt.getType()) ||
        !isa<cudaq::quake::WireType>(prev0Trgt.getType()) ||
        !isa<cudaq::quake::WireType>(prevTrgt.getType()) ||
        prev0Wires.empty() || prevWires.empty() || trgt != prev0Wires.back() ||
        prev0Trgt != prevWires.back()) {
      LLVM_DEBUG(llvm::dbgs() << "target wire must thread\n");
      return failure();
    }

    // Check that the controls (if any) are the same qubits.
    auto controls = qop.getControls();
    auto prev0Ctls = prev0.getControls();
    auto prevCtls = prev.getControls();
    if (controls.size() != prevCtls.size() ||
        prevCtls.size() != prev0Ctls.size()) {
      LLVM_DEBUG(llvm::dbgs() << "must have the same number of controls\n");
      return failure();
    }
    auto polarities = cudaq::opt::getControlPolarities(qop);
    if (polarities != cudaq::opt::getControlPolarities(prev0) ||
        polarities != cudaq::opt::getControlPolarities(prev)) {
      LLVM_DEBUG(llvm::dbgs() << "control polarities must be the same\n");
      return failure();
    }

    std::size_t prev0WireIndex = 0;
    std::size_t prevWireIndex = 0;
    for (auto [c, p0c, pc] : llvm::zip(controls, prev0Ctls, prevCtls)) {
      if (isa<cudaq::quake::ControlType>(c.getType())) {
        if (!isa<cudaq::quake::ControlType>(pc.getType()) || c != pc ||
            p0c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
        continue;
      }
      if (!isa<cudaq::quake::WireType>(c.getType()) ||
          !isa<cudaq::quake::WireType>(p0c.getType()) ||
          !isa<cudaq::quake::WireType>(pc.getType()) ||
          prev0WireIndex + 1 >= prev0Wires.size() ||
          prevWireIndex + 1 >= prevWires.size() ||
          c != prev0Wires[prev0WireIndex++] ||
          p0c != prevWires[prevWireIndex++]) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }
    if (prev0WireIndex + 1 != prev0Wires.size() ||
        prevWireIndex + 1 != prevWires.size())
      return failure();

    // X S Y = -S, while X S^dagger Y = S^dagger exactly.
    LLVM_DEBUG(llvm::dbgs() << "replaced: " << qop << '\n'
                            << prev0 << '\n'
                            << prev << '\n');
    SmallVector<Value> replacementControls(prevCtls);
    SmallVector<Value> replacementTargets(prevTrgs);
    auto replacement = cudaq::quake::SOp::create(
        rewriter, qop.getLoc(), qop.getResultTypes(), prev0.getIsAdjAttr(),
        ValueRange{}, replacementControls, replacementTargets,
        prev.getNegatedQubitControlsAttr());
    cudaq::opt::threadWireResults(replacement, replacementControls,
                                  replacementTargets);

    if (!prev0.isAdj()) {
      Value pi = cudaq::opt::factory::createPiConstant(qop.getLoc(), rewriter,
                                                       rewriter.getF64Type());
      auto correction = cudaq::opt::emitPhaseCorrection(
          rewriter, qop.getLoc(), pi, replacementControls,
          prev.getNegatedQubitControlsAttr(), replacementTargets.back());
      replacementControls = std::move(correction.controls);
      replacementTargets.back() = correction.anchor;
    }

    rewriter.replaceOp(qop, cudaq::opt::getWireValues(replacementControls,
                                                      replacementTargets));
    rewriter.eraseOp(prev0);
    rewriter.eraseOp(prev);
    ++stat;
    return success();
  }

private:
  Pass::Statistic &stat;
};

// A reset after a reset or a null_wire can be eliminated as it is redundant.
// NB: this optimization would not be valid after borrow_wire.
class EraseDoubleReset : public OpRewritePattern<cudaq::quake::ResetOp> {
public:
  EraseDoubleReset(MLIRContext *context, Pass::Statistic &stat)
      : OpRewritePattern(context), stat(stat) {}

  LogicalResult matchAndRewrite(cudaq::quake::ResetOp reset,
                                PatternRewriter &rewriter) const override {
    Value target = reset.getTargets();
    auto *ctx = rewriter.getContext();
    if (target.getType() != cudaq::quake::WireType::get(ctx))
      return failure();
    auto reset0 = target.template getDefiningOp<cudaq::quake::ResetOp>();
    if (reset0) {
      // Scalar def-use cannot exclude state changes through another SSA or
      // reference alias. Accept only an empty interval, conservatively missing
      // safe nonempty intervals.
      if (reset0->getNextNode() != reset.getOperation())
        return failure();
      if (shouldSkipRewrite(reset, reset0))
        return failure();
      LLVM_DEBUG(llvm::dbgs() << "eliminated: " << reset << '\n');
      rewriter.replaceOp(reset, reset0.getResults());
      ++stat;
      return success();
    }
    auto nullwire = target.template getDefiningOp<cudaq::quake::NullWireOp>();
    if (!nullwire) {
      LLVM_DEBUG(llvm::dbgs()
                 << "previous operation must be reset or null_wire\n");
      return failure();
    }
    if (shouldSkipRewrite(reset, nullwire))
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "eliminated: " << reset << '\n');
    rewriter.replaceOp(reset, nullwire.getResult());
    ++stat;
    return success();
  }

private:
  Pass::Statistic &stat;
};

static bool hasExclusiveNullWireLineage(Value wire, Operation *downstream) {
  Block *block = downstream->getBlock();
  while (Operation *producer = wire.getDefiningOp()) {
    if (producer->getBlock() != block || !wire.hasOneUse() ||
        wire.use_begin()->getOwner() != downstream)
      return false;
    if (isa<cudaq::quake::NullWireOp>(producer))
      return true;

    auto flow = cudaq::quake::detail::getScalarWireFlow(producer);
    if (!flow || flow->inputs.size() != 1 || flow->results.size() != 1 ||
        flow->results.front() != wire)
      return false;
    wire = flow->inputs.front();
    downstream = producer;
  }
  return false;
}

// Physical adjacency proves only that no operation lies between the reset and
// sink; reference and duplicate-wire aliases may still survive the sink.
// Restrict removal to an exclusive scalar lineage from a fresh local wire.
class EraseResetSink : public OpRewritePattern<cudaq::quake::SinkOp> {
public:
  EraseResetSink(MLIRContext *context, Pass::Statistic &stat)
      : OpRewritePattern(context), stat(stat) {}

  LogicalResult matchAndRewrite(cudaq::quake::SinkOp sink,
                                PatternRewriter &rewriter) const override {
    Value target = sink.getTarget();
    auto *ctx = rewriter.getContext();
    if (target.getType() != cudaq::quake::WireType::get(ctx))
      return failure();
    auto reset0 = target.template getDefiningOp<cudaq::quake::ResetOp>();
    if (!reset0) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must be reset\n");
      return failure();
    }
    // Keep the physical interval empty as a separate guard against effects
    // through aliases that do not appear on the scalar lineage.
    if (reset0->getNextNode() != sink.getOperation())
      return failure();
    if (!hasExclusiveNullWireLineage(reset0.getTargets(), reset0))
      return failure();
    if (shouldSkipRewrite(sink, reset0))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "eliminated: " << reset0 << '\n');
    rewriter.replaceOp(reset0, reset0.getTargets());
    ++stat;
    return success();
  }

private:
  Pass::Statistic &stat;
};

namespace {

class QuakeSimplifyPass
    : public cudaq::opt::impl::QuakeSimplifyBase<QuakeSimplifyPass> {
public:
  using QuakeSimplifyBase::QuakeSimplifyBase;

  void runOnOperation() override {
    if (getOperation()->hasAttr(cudaq::runtime::disableQuantumOpts))
      return;

    if (!std::isfinite(threshold) || threshold < 0.0 ||
        (rotationsToCliffordT &&
         (!std::isfinite(cliffordTEpsilon) || cliffordTEpsilon < 0.0))) {
      getOperation()->emitError(
          "quake-simplify requires non-negative finite thresholds");
      signalPassFailure();
      return;
    }

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    auto *ctx = &getContext();
    cudaq::opt::CommutationAwareRewriteDriver driver(*ctx, config);
    auto &patterns = driver.get_patterns();
    patterns.add<EraseDoubleReset, EraseResetSink>(ctx, numResetsErased);
    patterns.add<ReduceYSX>(ctx, numReduceYSXRewrites);
    auto &matcher = driver.get_matcher();

    patterns.add<PhasedRxCombine, RotationCombine<cudaq::quake::R1Op>,
                 RotationCombine<cudaq::quake::RxOp>,
                 RotationCombine<cudaq::quake::RyOp>,
                 RotationCombine<cudaq::quake::RzOp>>(
        ctx, matcher, threshold, numZeroRotationsEliminated,
        numRotationsCombined);
    patterns.add<DiscretePhaseFold<cudaq::quake::SOp, cudaq::quake::ZOp>>(
        ctx, matcher, numDoubleSRewrites);
    patterns.add<DiscretePhaseFold<cudaq::quake::TOp, cudaq::quake::SOp>>(
        ctx, matcher, numDoubleTRewrites);
    patterns.add<InverseElimination<cudaq::quake::HOp>,
                 InverseElimination<cudaq::quake::SwapOp>,
                 InverseElimination<cudaq::quake::XOp>,
                 InverseElimination<cudaq::quake::YOp>,
                 InverseElimination<cudaq::quake::ZOp>>(
        ctx, matcher, numHermitianEliminations);
    patterns.add<InverseElimination<cudaq::quake::SOp,
                                    InversePairKind::OppositeAdjoints>,
                 InverseElimination<cudaq::quake::TOp,
                                    InversePairKind::OppositeAdjoints>>(
        ctx, matcher, numAdjointEliminations);
    if (rotationsToCliffordT)
      populateRotationsToCliffordTPatterns(patterns, cliffordTEpsilon,
                                           numCliffordTRotations);

    if (failed(driver.run(getOperation()->getRegion(0))))
      signalPassFailure();
  }
};
} // namespace
