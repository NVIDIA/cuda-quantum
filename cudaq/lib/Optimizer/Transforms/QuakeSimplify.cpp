/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/CommutationAwareRewrite.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <cmath>
#include <type_traits>

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKESIMPLIFY
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "quake-simplify"

using namespace mlir;

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

// Splice both endpoints out of their wires. Each result denotes the same qubit
// as the operand it came from, so forwarding an operation's own wire operands
// to its own results is right whichever order it names those qubits in. The
// operations in between commute with the anchor and stay where they are.
template <typename QOP>
static void cancelPair(QOP anchor, QOP endpoint, PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "eliminated: " << anchor << '\n'
                          << endpoint << '\n');
  rewriter.replaceOp(endpoint, getWireOperands(endpoint));
  rewriter.replaceOp(anchor, getWireOperands(anchor));
}

// Cancel `anchor` against the nearest endpoint accepted by its gate family.
template <typename QOP, typename IsEndpoint>
static LogicalResult
cancelTransparentPair(QOP anchor,
                      cudaq::opt::CommutationAwareRewriteMatcher &matcher,
                      PatternRewriter &rewriter, IsEndpoint isEndpoint) {
  auto match = matcher.findNearest(
      anchor, cudaq::opt::CommutationSearchDirection::Forward, isEndpoint);
  if (!match)
    return failure();

  cancelPair(anchor, cast<QOP>(match->endpoint), rewriter);
  return success();
}

template <typename QOP>
class HermitianElimination : public OpRewritePattern<QOP> {
public:
  HermitianElimination(MLIRContext *context,
                       cudaq::opt::CommutationAwareRewriteMatcher &matcher,
                       Pass::Statistic &stat)
      : OpRewritePattern<QOP>(context), matcher(matcher), stat(stat) {}

  LogicalResult matchAndRewrite(QOP qop,
                                PatternRewriter &rewriter) const override {
    auto result = cancelTransparentPair(
        qop, matcher, rewriter, [&](Operation *candidate) {
          return static_cast<bool>(
              getSameActionEndpoint(qop, candidate, matcher));
        });
    if (succeeded(result))
      ++stat;
    return result;
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
  auto endpointWires = cudaq::quake::getWireOperands(endpoint);
  ValueRange anchorWires = anchor.getWires();
  if (endpointWires.size() != anchorWires.size() || anchorWires.size() < 2)
    return false;

  for (std::size_t i = 0, e = anchorWires.size() - 2; i != e; ++i)
    if (endpointWires[i] != anchorWires[i])
      return false;
  return endpointWires[endpointWires.size() - 2] == anchorWires.back() &&
         endpointWires.back() == anchorWires[anchorWires.size() - 2];
}

template <>
class HermitianElimination<cudaq::quake::SwapOp>
    : public OpRewritePattern<cudaq::quake::SwapOp> {
public:
  HermitianElimination(MLIRContext *context,
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
    if (succeeded(result))
      ++stat;
    return result;
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  Pass::Statistic &stat;
};

template <typename QOP>
class AdjointElimination : public OpRewritePattern<QOP> {
public:
  AdjointElimination(MLIRContext *context,
                     cudaq::opt::CommutationAwareRewriteMatcher &matcher,
                     Pass::Statistic &stat)
      : OpRewritePattern<QOP>(context), matcher(matcher), stat(stat) {}

  LogicalResult matchAndRewrite(QOP qop,
                                PatternRewriter &rewriter) const override {
    auto result = cancelTransparentPair(
        qop, matcher, rewriter, [&](Operation *candidate) {
          auto endpoint = getSameActionEndpoint(qop, candidate, matcher);
          return endpoint && qop.isAdj() != endpoint.isAdj();
        });
    if (succeeded(result))
      ++stat;
    return result;
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
// TODO: Consider if it would be better to add an I (identity) gate to Quake.
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
      rewriter.replaceOp(anchor, cudaq::quake::getWireOperands(anchor));
      ++zeroStat;
      return success();
    }

    // Stop at the first structurally matching action. Checking phi afterward
    // keeps a different axis as a barrier instead of searching past it.
    auto match = matcher.findNearest(
        anchor, cudaq::opt::CommutationSearchDirection::Forward,
        [&](Operation *candidate) {
          return static_cast<bool>(
              getSameActionEndpoint(anchor, candidate, matcher));
        });
    if (!match)
      return failure();

    auto endpoint = cast<cudaq::quake::PhasedRxOp>(match->endpoint);
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
    rewriter.replaceOp(anchor, cudaq::quake::getWireOperands(anchor));
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
      rewriter.replaceOp(anchor, cudaq::quake::getWireOperands(anchor));
      ++zeroStat;
      return success();
    }

    Value anchorAngle = parameters.front();
    auto match = matcher.findNearest(
        anchor, cudaq::opt::CommutationSearchDirection::Forward,
        [&](Operation *candidate) {
          auto endpoint = getSameActionEndpoint(anchor, candidate, matcher);
          if (!endpoint)
            return false;
          auto endpointParameters = endpoint.getParameters();
          return endpointParameters.size() == 1 &&
                 endpointParameters.front().getType() == anchorAngle.getType();
        });
    if (!match)
      return failure();

    auto endpoint = cast<QOP>(match->endpoint);
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
    rewriter.replaceOp(anchor, cudaq::quake::getWireOperands(anchor));
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
    auto match = matcher.findNearest(
        anchor, cudaq::opt::CommutationSearchDirection::Forward,
        [&](Operation *candidate) {
          return static_cast<bool>(
              getSameActionEndpoint(anchor, candidate, matcher));
        });
    if (!match)
      return failure();

    auto endpoint = cast<SourceOp>(match->endpoint);
    // Let AdjointElimination own the nearest opposite-adjoint pair.
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
    rewriter.replaceOp(anchor, getWireOperands(anchor));
    ++stat;
    return success();
  }

private:
  cudaq::opt::CommutationAwareRewriteMatcher &matcher;
  Pass::Statistic &stat;
};

// S = YSX
class ReduceYSX : public OpRewritePattern<cudaq::quake::XOp> {
public:
  ReduceYSX(MLIRContext *context, Pass::Statistic &stat)
      : OpRewritePattern(context), stat(stat) {}

  LogicalResult matchAndRewrite(cudaq::quake::XOp qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    // The uncontrolled rewrite is equal up to global phase. Under control,
    // that phase is relative between control branches and is observable.
    // TODO: Add explicit phase compensation when Quake models global phase.
    if (!qop.getControls().empty())
      return failure();

    auto targets = qop.getTargets();
    if (targets.size() != 1 ||
        !cudaq::quake::isQuantumValueType(targets[0].getType())) {
      LLVM_DEBUG(llvm::dbgs() << "operation must have 1 target\n");
      return failure();
    }
    Value trgt = targets[0];

    auto prev0 = targets[0].template getDefiningOp<cudaq::quake::SOp>();
    if (!prev0) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must be S\n");
      return failure();
    }
    auto prev =
        prev0.getTargets()[0].template getDefiningOp<cudaq::quake::YOp>();
    if (!prev) {
      LLVM_DEBUG(llvm::dbgs() << "previous previous operation must be Y\n");
      return failure();
    }
    if (prev0.getNegatedQubitControls() || prev.getNegatedQubitControls())
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
    for (auto iter :
         llvm::enumerate(llvm::zip(controls, prev0Ctls, prevCtls))) {
      auto n = iter.index();
      auto [c, p0c, pc] = iter.value();
      if (!isa<cudaq::quake::WireType>(c.getType()) ||
          !isa<cudaq::quake::WireType>(pc.getType()) ||
          !isa<cudaq::quake::WireType>(p0c.getType()) ||
          c != prev0.getResult(n) || p0c != prev.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

    // The uncontrolled Y-S-X product equals S up to global phase.
    LLVM_DEBUG(llvm::dbgs() << "replaced: " << qop << '\n'
                            << prev0 << '\n'
                            << prev << '\n');
    rewriter.replaceOpWithNewOp<cudaq::quake::SOp>(
        qop, qop.getResultTypes(), UnitAttr{}, ValueRange{}, prevCtls, prevTrgs,
        DenseBoolArrayAttr{});
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
    auto *ctx = &getContext();
    auto *op = getOperation();
    cudaq::opt::CommutationAwareRewriteDriver driver(*ctx);
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
    patterns.add<HermitianElimination<cudaq::quake::HOp>,
                 HermitianElimination<cudaq::quake::SwapOp>,
                 HermitianElimination<cudaq::quake::XOp>,
                 HermitianElimination<cudaq::quake::YOp>,
                 HermitianElimination<cudaq::quake::ZOp>>(
        ctx, matcher, numHermitianEliminations);
    patterns.add<AdjointElimination<cudaq::quake::SOp>,
                 AdjointElimination<cudaq::quake::TOp>>(ctx, matcher,
                                                        numAdjointEliminations);
    if (failed(driver.run(op->getRegion(0))))
      signalPassFailure();
  }
};
} // namespace
