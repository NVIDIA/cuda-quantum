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

// Nearest enclosing `cc.scope` carrying the `atomic_quantum_region` marker,
// skipping any ordinary scopes in between.
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

// Apply simple quantum optimizations to value-semantics Quake.
// Commutation-aware rewrites are supported for scalar wire controls and
// targets.

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

// Apply some simple quantum optimizations to quake. The quake operations are
// expected to be in the value-semantics (having wire or control type operands).

template <typename QOP>
static QOP
getSameActionEndpoint(QOP anchor, Operation *candidate,
                      cudaq::opt::CommutationAwareRewriteMatcher &matcher) {
  auto endpoint = dyn_cast<QOP>(candidate);
  if (!endpoint || !haveSameControlArityAndPolarity(anchor, endpoint) ||
      !matcher.haveSameOrderedQuantumOperands(anchor, endpoint))
    return {};
  return endpoint;
}

// Preserve the adjacent inverse path that predates commutation-aware search.
// A value defined outside an ordinary single-block cc.scope may be consumed
// directly inside it, even though the block-local matcher must stop at that
// boundary. Exact result-to-operand threading proves the ordered qubit roles;
// shareOptimizationRegion keeps atomic and unsupported boundaries opaque.
template <typename QOP>
static QOP getCrossScopePredecessor(QOP endpoint) {
  auto endpointFlow = cudaq::quake::detail::getScalarWireFlow(endpoint);
  if (!endpointFlow || endpointFlow->inputs.empty())
    return {};

  auto predecessor = endpointFlow->inputs.front().template getDefiningOp<QOP>();
  if (!predecessor || predecessor->getBlock() == endpoint->getBlock() ||
      shouldSkipRewrite(endpoint, predecessor) ||
      !haveSameControlArityAndPolarity(predecessor, endpoint))
    return {};

  return predecessor;
}

template <typename QOP>
static QOP getCrossScopeAdjacentPredecessor(QOP endpoint) {
  auto predecessor = getCrossScopePredecessor(endpoint);
  if (!predecessor)
    return {};

  auto endpointFlow = cudaq::quake::detail::getScalarWireFlow(endpoint);
  auto predecessorFlow = cudaq::quake::detail::getScalarWireFlow(predecessor);
  if (!endpointFlow || !predecessorFlow ||
      !llvm::equal(predecessorFlow->results, endpointFlow->inputs))
    return {};
  return predecessor;
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

// Splice both endpoints out of their wires. Each result denotes the same qubit
// as the operand it came from, so forwarding an operation's own wire operands
// to its own results is right whichever order it names those qubits in. The
// operations in between commute with the anchor and stay where they are.
template <typename QOP>
static void cancelPair(QOP anchor, QOP endpoint, PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "eliminated: " << anchor << '\n'
                          << endpoint << '\n');
  eraseOrForward(endpoint, rewriter);
  eraseOrForward(anchor, rewriter);
}

// Cancel `anchor` against the nearest endpoint accepted by its gate family.
template <typename QOP, typename IsEndpoint>
static LogicalResult
cancelTransparentPair(QOP anchor,
                      cudaq::opt::CommutationAwareRewriteMatcher &matcher,
                      PatternRewriter &rewriter, IsEndpoint isEndpoint) {
  Operation *endpoint = matcher.findNearest(anchor, isEndpoint);
  if (!endpoint)
    return failure();

  cancelPair(anchor, cast<QOP>(endpoint), rewriter);
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

  LogicalResult matchAndRewrite(QOP qop,
                                PatternRewriter &rewriter) const override {
    auto result = cancelTransparentPair(
        qop, matcher, rewriter, [&](Operation *candidate) {
          auto endpoint = getSameActionEndpoint(qop, candidate, matcher);
          return endpoint && (Kind == InversePairKind::SelfInverse ||
                              qop.isAdj() != endpoint.isAdj());
        });
    if (succeeded(result)) {
      ++stat;
      return success();
    }

    auto predecessor = getCrossScopeAdjacentPredecessor(qop);
    if (!predecessor || (Kind == InversePairKind::OppositeAdjoints &&
                         qop.isAdj() == predecessor.isAdj()))
      return failure();

    cancelPair(predecessor, qop, rewriter);
    ++stat;
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  Pass::Statistic &stat;
};

// Swap is symmetric in its two targets, so an endpoint naming them in the
// opposite order still inverts the anchor. The matcher's ordered-identity query
// cannot express that, so match the case positionally instead: each of the
// endpoint's wire operands must be the anchor's own results, with the final two
// target positions transposed. Linear use means this is necessarily adjacent.
static bool hasTransposedTargetsOnAnchorWires(cudaq::quake::SwapOp anchor,
                                              cudaq::quake::SwapOp endpoint) {
  auto anchorFlow = cudaq::quake::detail::getScalarWireFlow(anchor);
  auto endpointFlow = cudaq::quake::detail::getScalarWireFlow(endpoint);
  if (!anchorFlow || !endpointFlow)
    return false;
  const auto &endpointWires = endpointFlow->inputs;
  const auto &anchorWires = anchorFlow->results;
  if (endpointWires.size() != anchorWires.size() || anchorWires.size() < 2)
    return false;

  for (std::size_t i = 0, e = anchorWires.size() - 2; i != e; ++i)
    if (endpointWires[i] != anchorWires[i])
      return false;
  return endpointWires[endpointWires.size() - 2] == anchorWires.back() &&
         endpointWires.back() == anchorWires[anchorWires.size() - 2];
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

  LogicalResult matchAndRewrite(cudaq::quake::SwapOp qop,
                                PatternRewriter &rewriter) const override {
    auto result = cancelTransparentPair(
        qop, matcher, rewriter, [&](Operation *candidate) {
          auto endpoint = dyn_cast<cudaq::quake::SwapOp>(candidate);
          return endpoint && haveSameControlArityAndPolarity(qop, endpoint) &&
                 (matcher.haveSameOrderedQuantumOperands(qop, endpoint) ||
                  hasTransposedTargetsOnAnchorWires(qop, endpoint));
        });
    if (succeeded(result)) {
      ++stat;
      return success();
    }

    auto predecessor = getCrossScopePredecessor(qop);
    if (!predecessor || (!getCrossScopeAdjacentPredecessor(qop) &&
                         !hasTransposedTargetsOnAnchorWires(predecessor, qop)))
      return failure();

    cancelPair(predecessor, qop, rewriter);
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

// `quake.ry (12 * pi) %1` is a special backdoor NOP that is never optimized.
template <typename QOP>
static bool isBackdoorNopGate(double theta) {
  return std::is_same_v<QOP, cudaq::quake::RyOp> && theta == 12.0 * M_PI;
}

template <typename QOP>
static bool isIdentityRotation(QOP qop, double threshold) {
  Attribute attr;
  if (!matchPattern(qop.getParameters().front(), m_Constant(&attr)))
    return false;

  // A parameter may use any float type, so widen to double for the test.
  APFloat angle = cast<FloatAttr>(attr).getValue();
  bool lostPrecision = false;
  if (angle.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                    &lostPrecision) != APFloat::opOK ||
      lostPrecision)
    return false;
  double theta = angle.convertToDouble();

  if (isBackdoorNopGate<QOP>(theta))
    return false;

  // Never the shorter period: the `-1` an axis rotation picks up at `2*pi` is
  // observable if the rotation runs under a control, and this op may yet be
  // given one by control synthesis of the function it is in.
  double residual = std::remainder(theta, exactIdentityPeriod<QOP>());

  // At its default the threshold admits only representation error.
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
                                         Location location, Value anchorAngle,
                                         bool anchorIsAdjoint,
                                         Value endpointAngle,
                                         bool endpointIsAdjoint) {
  Type angleType = endpointAngle.getType();
  if (endpointIsAdjoint)
    endpointAngle =
        arith::NegFOp::create(rewriter, location, angleType, endpointAngle);
  if (anchorIsAdjoint)
    anchorAngle =
        arith::NegFOp::create(rewriter, location, angleType, anchorAngle);
  return arith::AddFOp::create(rewriter, location, angleType, endpointAngle,
                               anchorAngle);
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

  LogicalResult matchAndRewrite(cudaq::quake::PhasedRxOp anchor,
                                PatternRewriter &rewriter) const override {
    auto anchorParameters = anchor.getParameters();
    if (anchorParameters.size() != 2)
      return failure();

    if (isIdentityRotation(anchor, threshold)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "zero rotation eliminated [" << anchor << "]\n");
      eraseOrForward(anchor, rewriter);
      ++zeroStat;
      return success();
    }

    // Stop at the first structurally matching action. Checking phi afterward
    // keeps a different axis as a barrier instead of searching past it.
    Operation *match = matcher.findNearest(anchor, [&](Operation *candidate) {
      return static_cast<bool>(
          getSameActionEndpoint(anchor, candidate, matcher));
    });
    if (!match)
      return failure();

    auto endpoint = cast<cudaq::quake::PhasedRxOp>(match);
    auto endpointParameters = endpoint.getParameters();
    if (endpointParameters.size() != 2 ||
        !haveExactValue(anchorParameters[1], endpointParameters[1]) ||
        anchorParameters.front().getType() !=
            endpointParameters.front().getType())
      return failure();

    Value anchorTheta = anchorParameters.front();
    Value endpointTheta = endpointParameters.front();
    if (anchor.isAdj() != endpoint.isAdj() &&
        haveExactValue(anchorTheta, endpointTheta)) {
      cancelPair(anchor, endpoint, rewriter);
      ++combineStat;
      return success();
    }

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(endpoint);
      Value combinedTheta = createCombinedRotationAngle(
          rewriter, endpoint.getLoc(), anchorTheta, anchor.isAdj(),
          endpointTheta, endpoint.isAdj());

      LLVM_DEBUG(llvm::dbgs() << "combined: " << anchor << '\n'
                              << endpoint << '\n');
      [[maybe_unused]] auto combined =
          rewriter.replaceOpWithNewOp<cudaq::quake::PhasedRxOp>(
              endpoint, endpoint.getResultTypes(), UnitAttr{},
              ValueRange{combinedTheta, endpointParameters[1]},
              endpoint.getControls(), endpoint.getTargets(),
              endpoint.getNegatedQubitControlsAttr());
      LLVM_DEBUG(llvm::dbgs() << "into: " << combined << '\n');
    }
    eraseOrForward(anchor, rewriter);
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

  LogicalResult matchAndRewrite(QOP anchor,
                                PatternRewriter &rewriter) const override {
    auto parameters = anchor.getParameters();
    if (parameters.size() != 1)
      return failure();

    if (isIdentityRotation(anchor, threshold)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "zero rotation eliminated [" << anchor << "]\n");
      eraseOrForward(anchor, rewriter);
      ++zeroStat;
      return success();
    }

    Value anchorAngle = parameters.front();
    Operation *match = matcher.findNearest(anchor, [&](Operation *candidate) {
      auto endpoint = getSameActionEndpoint(anchor, candidate, matcher);
      if (!endpoint)
        return false;
      auto endpointParameters = endpoint.getParameters();
      return endpointParameters.size() == 1 &&
             endpointParameters.front().getType() == anchorAngle.getType();
    });
    if (!match)
      return failure();

    auto endpoint = cast<QOP>(match);
    Value endpointAngle = endpoint.getParameters().front();
    if (anchor.isAdj() != endpoint.isAdj() &&
        haveExactValue(anchorAngle, endpointAngle)) {
      cancelPair(anchor, endpoint, rewriter);
      ++combineStat;
      return success();
    }

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(endpoint);
      Value combinedAngle = createCombinedRotationAngle(
          rewriter, endpoint.getLoc(), anchorAngle, anchor.isAdj(),
          endpointAngle, endpoint.isAdj());

      LLVM_DEBUG(llvm::dbgs() << "combined: " << anchor << '\n'
                              << endpoint << '\n');
      [[maybe_unused]] auto combined = rewriter.replaceOpWithNewOp<QOP>(
          endpoint, endpoint.getResultTypes(), UnitAttr{},
          ValueRange{combinedAngle}, endpoint.getControls(),
          endpoint.getTargets(), endpoint.getNegatedQubitControlsAttr());
      LLVM_DEBUG(llvm::dbgs() << "into: " << combined << '\n');
    }
    eraseOrForward(anchor, rewriter);
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

  LogicalResult matchAndRewrite(SourceOp anchor,
                                PatternRewriter &rewriter) const override {
    Operation *match = matcher.findNearest(anchor, [&](Operation *candidate) {
      return static_cast<bool>(
          getSameActionEndpoint(anchor, candidate, matcher));
    });
    if (!match)
      return failure();

    auto endpoint = cast<SourceOp>(match);
    // Let the inverse-elimination pattern own the opposite-adjoint pair.
    if (anchor.isAdj() != endpoint.isAdj())
      return failure();

    UnitAttr foldedAdjoint;
    if constexpr (std::is_same_v<FoldedOp, cudaq::quake::SOp>)
      foldedAdjoint = endpoint.getIsAdjAttr();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(endpoint);
      rewriter.replaceOpWithNewOp<FoldedOp>(
          endpoint, endpoint.getResultTypes(), foldedAdjoint, ValueRange{},
          endpoint.getControls(), endpoint.getTargets(),
          endpoint.getNegatedQubitControlsAttr());
    }
    eraseOrForward(anchor, rewriter);
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

    auto prev0 = targets.front().template getDefiningOp<cudaq::quake::SOp>();
    if (!prev0 || !prev0.getControls().empty() ||
        prev0.getTargets().size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must be S\n");
      return failure();
    }
    auto prev =
        prev0.getTargets().front().template getDefiningOp<cudaq::quake::YOp>();
    if (!prev || !prev.getControls().empty() || prev.getTargets().size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "previous previous operation must be Y\n");
      return failure();
    }

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
    auto last0 = prev0.getNumResults() - 1;
    auto last = prev.getNumResults() - 1;
    if (!isa<cudaq::quake::WireType>(trgt.getType()) ||
        !isa<cudaq::quake::WireType>(prev0Trgt.getType()) ||
        !isa<cudaq::quake::WireType>(prevTrgt.getType()) ||
        trgt != prev0.getResult(last0) || prev0Trgt != prev.getResult(last)) {
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

    for (auto iter :
         llvm::enumerate(llvm::zip(controls, prev0Ctls, prevCtls))) {
      auto n = iter.index();
      auto [c, p0c, pc] = iter.value();
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
          c != prev0.getResult(n) || p0c != prev.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

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

// A reset before a sink can be eliminated as the wire is going out of scope.
// NB: this optimization would not be valid before return_wire.
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
    auto &patterns = driver.getPatterns();
    patterns.add<EraseDoubleReset, EraseResetSink>(ctx, numResetsErased);
    patterns.add<ReduceYSX>(ctx, numReduceYSXRewrites);
    auto &matcher = driver.getMatcher();

    // Combine rotations, including phased rotations.
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
