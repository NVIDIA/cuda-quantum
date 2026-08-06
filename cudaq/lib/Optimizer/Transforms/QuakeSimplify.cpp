/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <cmath>

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKESIMPLIFY
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "quake-simplify"

using namespace mlir;

template <typename C>
void filterArgs(SmallVector<Value> &args, C collection) {
  for (auto item : collection)
    if (cudaq::quake::isQuantumValueType(item.getType()))
      args.push_back(item);
}

// Apply some simple quantum optimizations to quake. The quake operations are
// expected to be in the value-semantics (having wire or control type operands).

template <typename QOP>
class HermitianElimination : public OpRewritePattern<QOP> {
public:
  using Base = OpRewritePattern<QOP>;

  HermitianElimination(MLIRContext *ctx, Pass::Statistic &stat)
      : Base(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(QOP qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    auto targets = qop.getTargets();
    if (targets.size() != 1 ||
        !cudaq::quake::isQuantumValueType(targets[0].getType())) {
      LLVM_DEBUG(llvm::dbgs() << "operation must have 1 target\n");
      return failure();
    }
    Value trgt = targets[0];

    // Check that these are the same Hermitian op back-to-back.
    auto prev = targets[0].template getDefiningOp<QOP>();
    if (!prev) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must be the same\n");
      return failure();
    }
    if (prev.getNegatedQubitControls())
      return failure();

    // Check target is properly threaded.
    auto prevTrgs = prev.getTargets();
    if (prevTrgs.size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must have 1 target\n");
      return failure();
    }
    Value prevTrgt = prevTrgs[0];
    auto last = prev.getNumResults() - 1;
    if (!isa<cudaq::quake::WireType>(trgt.getType()) ||
        !isa<cudaq::quake::WireType>(prevTrgt.getType()) ||
        trgt != prev.getResult(last)) {
      LLVM_DEBUG(llvm::dbgs() << "target wire must thread\n");
      return failure();
    }

    // Check that the controls (if any) are the same qubits.
    auto controls = qop.getControls();
    auto prevCtls = prev.getControls();
    if (controls.size() != prevCtls.size()) {
      LLVM_DEBUG(llvm::dbgs() << "must have the same number of controls\n");
      return failure();
    }
    for (auto iter : llvm::enumerate(llvm::zip(controls, prevCtls))) {
      auto n = iter.index();
      auto [c, pc] = iter.value();
      if (isa<cudaq::quake::ControlType>(c.getType()))
        if (!isa<cudaq::quake::ControlType>(pc.getType()) || c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<cudaq::quake::WireType>(c.getType()) ||
          !isa<cudaq::quake::WireType>(pc.getType()) ||
          c != prev.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

    // Eliminate the back-to-back Hermitian gates.
    SmallVector<Value> newOperands;
    filterArgs(newOperands, prevCtls);
    filterArgs(newOperands, prevTrgs);
    LLVM_DEBUG(llvm::dbgs() << "eliminated: " << qop << '\n' << prev << '\n');
    rewriter.replaceOp(qop, newOperands);
    rewriter.eraseOp(prev);
    ++stat;
    return success();
  }

private:
  Pass::Statistic &stat;
};

template <>
class HermitianElimination<cudaq::quake::SwapOp>
    : public OpRewritePattern<cudaq::quake::SwapOp> {
public:
  using Base = OpRewritePattern<cudaq::quake::SwapOp>;

  HermitianElimination(MLIRContext *ctx, Pass::Statistic &stat)
      : Base(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(cudaq::quake::SwapOp qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    auto targets = qop.getTargets();
    if (targets.size() != 2 ||
        !cudaq::quake::isQuantumValueType(targets[0].getType()) ||
        !cudaq::quake::isQuantumValueType(targets[1].getType())) {
      LLVM_DEBUG(llvm::dbgs() << "operation must have 2 targets\n");
      return failure();
    }

    // Check that these are the same swap op back-to-back.
    auto prev0 = targets[0].template getDefiningOp<cudaq::quake::SwapOp>();
    if (!prev0) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation 0 must be the same\n");
      return failure();
    }
    auto prev1 = targets[1].template getDefiningOp<cudaq::quake::SwapOp>();
    if (!prev1) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation 1 must be the same\n");
      return failure();
    }
    if (prev0 != prev1) {
      LLVM_DEBUG(llvm::dbgs() << "previous operations must be the same\n");
      return failure();
    }
    if (prev0.getNegatedQubitControls())
      return failure();

    // Check target is properly threaded.
    auto prevTrgs = prev0.getTargets();
    if (prevTrgs.size() != 2) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must have 2 target\n");
      return failure();
    }
    auto last = prev0.getNumResults() - 1;
    auto matches = [](Value u0, Value u1, Value d0, Value d1) -> bool {
      return (u0 == d0 && u1 == d1) || (u0 == d1 && u1 == d0);
    };
    if (!isa<cudaq::quake::WireType>(targets[0].getType()) ||
        !isa<cudaq::quake::WireType>(prevTrgs[0].getType()) ||
        !isa<cudaq::quake::WireType>(targets[1].getType()) ||
        !isa<cudaq::quake::WireType>(prevTrgs[1].getType()) ||
        !matches(targets[0], targets[1], prev0.getResult(last - 1),
                 prev0.getResult(last))) {
      LLVM_DEBUG(llvm::dbgs() << "target wires must thread\n");
      return failure();
    }

    // Check that the controls (if any) are the same qubits.
    auto controls = qop.getControls();
    auto prevCtls = prev0.getControls();
    if (controls.size() != prevCtls.size()) {
      LLVM_DEBUG(llvm::dbgs() << "must have the same number of controls\n");
      return failure();
    }
    for (auto iter : llvm::enumerate(llvm::zip(controls, prevCtls))) {
      auto n = iter.index();
      auto [c, pc] = iter.value();
      if (isa<cudaq::quake::ControlType>(c.getType()))
        if (!isa<cudaq::quake::ControlType>(pc.getType()) || c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<cudaq::quake::WireType>(c.getType()) ||
          !isa<cudaq::quake::WireType>(pc.getType()) ||
          c != prev0.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

    // Eliminate the back-to-back Hermitian swap gates.
    SmallVector<Value> newOperands;
    filterArgs(newOperands, prevCtls);
    filterArgs(newOperands, prevTrgs);
    LLVM_DEBUG(llvm::dbgs() << "eliminated: " << qop << '\n' << prev0 << '\n');
    rewriter.replaceOp(qop, newOperands);
    rewriter.eraseOp(prev0);
    ++stat;
    return success();
  }

private:
  Pass::Statistic &stat;
};

template <typename QOP>
class AdjointElimination : public OpRewritePattern<QOP> {
public:
  using Base = OpRewritePattern<QOP>;

  AdjointElimination(MLIRContext *ctx, Pass::Statistic &stat)
      : Base(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(QOP qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    auto targets = qop.getTargets();
    if (targets.size() != 1 ||
        !cudaq::quake::isQuantumValueType(targets[0].getType())) {
      LLVM_DEBUG(llvm::dbgs() << "operation must have 1 target\n");
      return failure();
    }
    Value trgt = targets[0];

    // Check that these are the same op back-to-back.
    auto prev = targets[0].template getDefiningOp<QOP>();
    if (!prev) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must be the same class\n");
      return failure();
    }
    if (prev.getNegatedQubitControls())
      return failure();

    // If the two are not converse in their adjoint setting, nothing to do.
    if (qop.isAdj() == prev.isAdj()) {
      LLVM_DEBUG(llvm::dbgs() << "operations [" << qop << ", " << prev
                              << "] are not adjoint inverses\n");
      return failure();
    }

    // Check target is properly threaded.
    auto prevTrgs = prev.getTargets();
    if (prevTrgs.size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must have 1 target\n");
      return failure();
    }
    Value prevTrgt = prevTrgs[0];
    auto last = prev.getNumResults() - 1;
    if (!isa<cudaq::quake::WireType>(trgt.getType()) ||
        !isa<cudaq::quake::WireType>(prevTrgt.getType()) ||
        trgt != prev.getResult(last)) {
      LLVM_DEBUG(llvm::dbgs() << "target wire must thread\n");
      return failure();
    }

    // Check that the controls (if any) are the same qubits.
    auto controls = qop.getControls();
    auto prevCtls = prev.getControls();
    if (controls.size() != prevCtls.size()) {
      LLVM_DEBUG(llvm::dbgs() << "must have the same number of controls\n");
      return failure();
    }
    for (auto iter : llvm::enumerate(llvm::zip(controls, prevCtls))) {
      auto n = iter.index();
      auto [c, pc] = iter.value();
      if (isa<cudaq::quake::ControlType>(c.getType()))
        if (!isa<cudaq::quake::ControlType>(pc.getType()) || c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<cudaq::quake::WireType>(c.getType()) ||
          !isa<cudaq::quake::WireType>(pc.getType()) ||
          c != prev.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

    // Eliminate the back-to-back gates.
    SmallVector<Value> newOperands;
    filterArgs(newOperands, prevCtls);
    filterArgs(newOperands, prevTrgs);
    LLVM_DEBUG(llvm::dbgs() << "eliminated: " << qop << '\n' << prev << '\n');
    rewriter.replaceOp(qop, newOperands);
    rewriter.eraseOp(prev);
    ++stat;
    return success();
  }

private:
  Pass::Statistic &stat;
};

template <typename QOP>
class RotationCombine : public OpRewritePattern<QOP> {
public:
  using Base = OpRewritePattern<QOP>;

  RotationCombine(MLIRContext *ctx, double threshold, Pass::Statistic &zeroStat,
                  Pass::Statistic &combineStat)
      : Base(ctx), threshold(threshold), zeroStat(zeroStat),
        combineStat(combineStat) {}

  LogicalResult matchAndRewrite(QOP qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    if (isIdentityRotation(qop)) {
      // Forward the target to the uses.
      LLVM_DEBUG(llvm::dbgs() << "zero rotation eliminated [" << qop << "]\n");
      SmallVector<Value> newOperands;
      filterArgs(newOperands, qop.getControls());
      filterArgs(newOperands, qop.getTargets());

      rewriter.replaceOp(qop, newOperands);
      ++zeroStat;
      return success();
    }

    auto targets = qop.getTargets();
    if (targets.size() != 1 ||
        !cudaq::quake::isQuantumValueType(targets[0].getType())) {
      LLVM_DEBUG(llvm::dbgs() << "must have 1 target\n");
      return failure();
    }
    Value trgt = targets[0];

    // Check that these are the same rotation op back-to-back.
    auto prev = targets[0].template getDefiningOp<QOP>();
    if (!prev) {
      LLVM_DEBUG(llvm::dbgs() << "previous op must be the same\n"
                              << qop << '\n');
      return failure();
    }
    if (prev.getNegatedQubitControls())
      return failure();

    // Check target is properly threaded.
    auto prevTrgs = prev.getTargets();
    if (prevTrgs.size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "previous op must have 1 target\n");
      return failure();
    }
    Value prevTrgt = prevTrgs[0];
    auto last = prev.getNumResults() - 1;
    if (!isa<cudaq::quake::WireType>(trgt.getType()) ||
        !isa<cudaq::quake::WireType>(prevTrgt.getType()) ||
        trgt != prev.getResult(last)) {
      LLVM_DEBUG(llvm::dbgs() << "target wire must thread\n" << qop << '\n');
      return failure();
    }

    // Check that the controls (if any) are the same qubits.
    auto controls = qop.getControls();
    auto prevCtls = prev.getControls();
    if (controls.size() != prevCtls.size()) {
      LLVM_DEBUG(llvm::dbgs() << "must have the same number of controls\n");
      return failure();
    }
    for (auto iter : llvm::enumerate(llvm::zip(controls, prevCtls))) {
      auto n = iter.index();
      auto [c, pc] = iter.value();
      if (isa<cudaq::quake::ControlType>(c.getType()))
        if (!isa<cudaq::quake::ControlType>(pc.getType()) || c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<cudaq::quake::WireType>(c.getType()) ||
          !isa<cudaq::quake::WireType>(pc.getType()) ||
          c != prev.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control must be threaded\n");
        return failure();
      }
    }

    SmallVector<Value> params = qop.getParameters();
    SmallVector<Value> prevParams = prev.getParameters();
    if (params.size() != prevParams.size()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Two identical ops with different numbers of parameters?\n"
                 << qop << '\n'
                 << prev << '\n');
      return failure(); // This should never happen.
    }

    // Compute the new parameters. Negate all if adjoint is set.
    SmallVector<Value> newParams;
    auto loc = qop.getLoc();
    for (auto [p, pp] : llvm::zip(params, prevParams)) {
      auto ty = p.getType();
      if (ty != pp.getType()) {
        LLVM_DEBUG(llvm::dbgs() << "parameters must have same type\n");
        return failure();
      }
      if (qop.isAdj())
        p = arith::NegFOp::create(rewriter, loc, ty, p);
      if (prev.isAdj())
        pp = arith::NegFOp::create(rewriter, loc, ty, pp);
      newParams.push_back(arith::AddFOp::create(rewriter, loc, ty, p, pp));
    }

    // Combine the two rotations.
    LLVM_DEBUG(llvm::dbgs() << "combined: " << qop << '\n' << prev << '\n');
    [[maybe_unused]] auto newOp = rewriter.replaceOpWithNewOp<QOP>(
        qop, qop.getResultTypes(), UnitAttr{}, newParams, prevCtls, prevTrgs,
        DenseBoolArrayAttr{});
    rewriter.eraseOp(prev);
    LLVM_DEBUG(llvm::dbgs() << "into: " << newOp << '\n');
    ++combineStat;
    return success();
  }

private:
  // The angle is folded into the period after which the op is the identity,
  // so a full turn is caught along with a zero angle.
  bool isIdentityRotation(QOP qop) const {
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

    if (isBackdoorNopGate(theta))
      return false;

    // Never the shorter period: the `-1` an axis rotation picks up at `2*pi` is
    // observable if the rotation runs under a control, and this op may yet be
    // given one by control synthesis of the function it is in.
    double residual = std::remainder(theta, exactIdentityPeriod());

    // At its default the threshold admits only representation error.
    return std::abs(residual) <= threshold;
  }

  /// `quake.ry (12 * π) %1` is a special backdoor NOP that is never optimized.
  /// TODO: Consider if it would be better to add an I (identity) gate to Quake.
  static bool isBackdoorNopGate(double theta) {
    if constexpr (std::same_as<QOP, cudaq::quake::RyOp>) {
      return theta == 12.0 * M_PI;
    } else {
      return false;
    }
  }

  // The angle, in radians, after which QOP is exactly the identity operator.
  // The axis rotations are spinors: they pick up an overall `-1` at `2*pi` and
  // return to the identity only after `4*pi`. `r1` is `diag(1, exp(i*theta))`
  // and has period `2*pi` outright.
  static constexpr double exactIdentityPeriod() {
    if constexpr (std::is_same_v<QOP, cudaq::quake::R1Op>) {
      return 2.0 * M_PI;
    } else {
      return 4.0 * M_PI;
    }
  }

  double threshold;
  Pass::Statistic &zeroStat;
  Pass::Statistic &combineStat;
};

// Z = SS = S<adj>S<adj>
// I = SS<adj> = S<adj>S
class DoubleSOp : public OpRewritePattern<cudaq::quake::SOp> {
public:
  DoubleSOp(MLIRContext *ctx, Pass::Statistic &stat)
      : OpRewritePattern(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(cudaq::quake::SOp qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    auto targets = qop.getTargets();
    if (targets.size() != 1 ||
        !cudaq::quake::isQuantumValueType(targets[0].getType())) {
      LLVM_DEBUG(llvm::dbgs() << "operation must have 1 target\n");
      return failure();
    }
    Value trgt = targets[0];

    // Check that these are the same op back-to-back.
    auto prev = targets[0].template getDefiningOp<cudaq::quake::SOp>();
    if (!prev) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must be the same\n");
      return failure();
    }
    if (prev.getNegatedQubitControls())
      return failure();
    if (qop.isAdj() != prev.isAdj()) {
      LLVM_DEBUG(llvm::dbgs() << "operations have converse adjoint\n");
      return failure();
    }

    // Check target is properly threaded.
    auto prevTrgs = prev.getTargets();
    if (prevTrgs.size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must have 1 target\n");
      return failure();
    }
    Value prevTrgt = prevTrgs[0];
    auto last = prev.getNumResults() - 1;
    if (!isa<cudaq::quake::WireType>(trgt.getType()) ||
        !isa<cudaq::quake::WireType>(prevTrgt.getType()) ||
        trgt != prev.getResult(last)) {
      LLVM_DEBUG(llvm::dbgs() << "target wire must thread\n");
      return failure();
    }

    // Check that the controls (if any) are the same qubits.
    auto controls = qop.getControls();
    auto prevCtls = prev.getControls();
    if (controls.size() != prevCtls.size()) {
      LLVM_DEBUG(llvm::dbgs() << "must have the same number of controls\n");
      return failure();
    }
    for (auto iter : llvm::enumerate(llvm::zip(controls, prevCtls))) {
      auto n = iter.index();
      auto [c, pc] = iter.value();
      if (isa<cudaq::quake::ControlType>(c.getType()))
        if (!isa<cudaq::quake::ControlType>(pc.getType()) || c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<cudaq::quake::WireType>(c.getType()) ||
          !isa<cudaq::quake::WireType>(pc.getType()) ||
          c != prev.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

    if (qop.isAdj() != prev.isAdj()) {
      // Opposite adjoints cancel. Forward the wires entering the pair to users
      // of the second operation, then erase the first operation.
      SmallVector<Value> replacementWires;
      replacementWires.append(prevCtls.begin(), prevCtls.end());
      replacementWires.append(prevTrgs.begin(), prevTrgs.end());
      rewriter.replaceOp(qop, replacementWires);
      rewriter.eraseOp(prev);
      return success();
    }

    // Rewrite the back-to-back S gates.
    LLVM_DEBUG(llvm::dbgs() << "replaced: " << qop << '\n' << prev << '\n');
    rewriter.replaceOpWithNewOp<cudaq::quake::ZOp>(
        qop, qop.getResultTypes(), UnitAttr{}, ValueRange{}, prevCtls, prevTrgs,
        DenseBoolArrayAttr{});
    rewriter.eraseOp(prev);
    ++stat;
    return success();
  }

private:
  Pass::Statistic &stat;
};

// S = TT
// S<adj> = T<adj>T<adj>
// I = TT<adj> = T<adj>T
class DoubleTOp : public OpRewritePattern<cudaq::quake::TOp> {
public:
  DoubleTOp(MLIRContext *ctx, Pass::Statistic &stat)
      : OpRewritePattern(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(cudaq::quake::TOp qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    auto targets = qop.getTargets();
    if (targets.size() != 1 ||
        !cudaq::quake::isQuantumValueType(targets[0].getType())) {
      LLVM_DEBUG(llvm::dbgs() << "operation must have 1 target\n");
      return failure();
    }
    Value trgt = targets[0];

    // Check that these are the same op back-to-back.
    auto prev = targets[0].template getDefiningOp<cudaq::quake::TOp>();
    if (!prev) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must be T\n");
      return failure();
    }
    if (prev.getNegatedQubitControls())
      return failure();
    if (qop.isAdj() != prev.isAdj()) {
      LLVM_DEBUG(llvm::dbgs() << "operations have converse adjoint\n");
      return failure();
    }

    // Check target is properly threaded.
    auto prevTrgs = prev.getTargets();
    if (prevTrgs.size() != 1) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must have 1 target\n");
      return failure();
    }
    Value prevTrgt = prevTrgs[0];
    auto last = prev.getNumResults() - 1;
    if (!isa<cudaq::quake::WireType>(trgt.getType()) ||
        !isa<cudaq::quake::WireType>(prevTrgt.getType()) ||
        trgt != prev.getResult(last)) {
      LLVM_DEBUG(llvm::dbgs() << "target wire must thread\n");
      return failure();
    }

    // Check that the controls (if any) are the same qubits.
    auto controls = qop.getControls();
    auto prevCtls = prev.getControls();
    if (controls.size() != prevCtls.size()) {
      LLVM_DEBUG(llvm::dbgs() << "must have the same number of controls\n");
      return failure();
    }
    for (auto iter : llvm::enumerate(llvm::zip(controls, prevCtls))) {
      auto n = iter.index();
      auto [c, pc] = iter.value();
      if (isa<cudaq::quake::ControlType>(c.getType()))
        if (!isa<cudaq::quake::ControlType>(pc.getType()) || c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<cudaq::quake::WireType>(c.getType()) ||
          !isa<cudaq::quake::WireType>(pc.getType()) ||
          c != prev.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

    if (qop.isAdj() != prev.isAdj()) {
      // Opposite adjoints cancel. Forward the wires entering the pair to users
      // of the second operation, then erase the first operation.
      SmallVector<Value> replacementWires;
      replacementWires.append(prevCtls.begin(), prevCtls.end());
      replacementWires.append(prevTrgs.begin(), prevTrgs.end());
      rewriter.replaceOp(qop, replacementWires);
      rewriter.eraseOp(prev);
      return success();
    }

    // Rewrite the back-to-back S gates.
    LLVM_DEBUG(llvm::dbgs() << "replaced: " << qop << '\n' << prev << '\n');
    rewriter.replaceOpWithNewOp<cudaq::quake::SOp>(
        qop, qop.getResultTypes(), qop.getIsAdjAttr(), ValueRange{}, prevCtls,
        prevTrgs, DenseBoolArrayAttr{});
    rewriter.eraseOp(prev);
    ++stat;
    return success();
  }

private:
  Pass::Statistic &stat;
};

// S = YSX
class ReduceYSX : public OpRewritePattern<cudaq::quake::XOp> {
public:
  ReduceYSX(MLIRContext *ctx, Pass::Statistic &stat)
      : OpRewritePattern(ctx), stat(stat) {}

  LogicalResult matchAndRewrite(cudaq::quake::XOp qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    // The uncontrolled rewrite is equal up to global phase. Under control,
    // that phase is relative between control branches and is observable.
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
      if (isa<cudaq::quake::ControlType>(c.getType()))
        if (!isa<cudaq::quake::ControlType>(pc.getType()) || c != pc ||
            p0c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<cudaq::quake::WireType>(c.getType()) ||
          !isa<cudaq::quake::WireType>(pc.getType()) ||
          !isa<cudaq::quake::WireType>(pc.getType()) ||
          c != prev0.getResult(n) || p0c != prev.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

    // Rewrite the back-to-back S gates.
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
  EraseDoubleReset(MLIRContext *ctx, Pass::Statistic &stat)
      : OpRewritePattern(ctx), stat(stat) {}

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
  EraseResetSink(MLIRContext *ctx, Pass::Statistic &stat)
      : OpRewritePattern(ctx), stat(stat) {}

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
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    RewritePatternSet patterns(ctx);
    patterns.add<HermitianElimination<cudaq::quake::HOp>,
                 HermitianElimination<cudaq::quake::SwapOp>,
                 HermitianElimination<cudaq::quake::XOp>,
                 HermitianElimination<cudaq::quake::YOp>,
                 HermitianElimination<cudaq::quake::ZOp>>(
        ctx, numHermitianEliminations);
    patterns.add<AdjointElimination<cudaq::quake::SOp>,
                 AdjointElimination<cudaq::quake::TOp>>(ctx,
                                                        numAdjointEliminations);
    patterns.add<DoubleSOp>(ctx, numDoubleSRewrites);
    patterns.add<DoubleTOp>(ctx, numDoubleTRewrites);
    patterns.add<EraseDoubleReset, EraseResetSink>(ctx, numResetsErased);
    patterns.add<ReduceYSX>(ctx, numReduceYSXRewrites);
    patterns.add<RotationCombine<cudaq::quake::R1Op>,
                 RotationCombine<cudaq::quake::RxOp>,
                 RotationCombine<cudaq::quake::RyOp>,
                 RotationCombine<cudaq::quake::RzOp>,
                 RotationCombine<cudaq::quake::PhasedRxOp>>(
        ctx, threshold, numZeroRotationsEliminated, numRotationsCombined);
    if (failed(applyPatternsGreedily(op, std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace
