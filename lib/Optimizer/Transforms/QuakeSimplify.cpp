/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKESIMPLIFY
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "quake-simplify"

using namespace mlir;

// Apply some simple quantum optimizations to quake. The quake operations are
// expected to be in the value-semantics (having wire or control type operands).

template <typename QOP>
class HermitianElimination : public OpRewritePattern<QOP> {
public:
  using Base = OpRewritePattern<QOP>;
  using Base::Base;

  LogicalResult matchAndRewrite(QOP qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    auto targets = qop.getTargets();
    if (targets.size() != 1 ||
        !quake::isQuantumValueType(targets[0].getType())) {
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
    if (!isa<quake::WireType>(trgt.getType()) ||
        !isa<quake::WireType>(prevTrgt.getType()) ||
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
      if (isa<quake::ControlType>(c.getType()))
        if (!isa<quake::ControlType>(pc.getType()) || c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<quake::WireType>(c.getType()) ||
          !isa<quake::WireType>(pc.getType()) || c != prev.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

    // Eliminate the back-to-back Hermitian gates.
    SmallVector<Value> newOperands;
    newOperands.append(prevCtls.begin(), prevCtls.end());
    newOperands.append(prevTrgs.begin(), prevTrgs.end());
    LLVM_DEBUG(llvm::dbgs() << "eliminated: " << qop << '\n' << prev << '\n');
    rewriter.replaceOp(qop, newOperands);
    rewriter.eraseOp(prev);
    return success();
  }
};

template <>
class HermitianElimination<quake::SwapOp>
    : public OpRewritePattern<quake::SwapOp> {
public:
  using Base = OpRewritePattern<quake::SwapOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(quake::SwapOp qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    auto targets = qop.getTargets();
    if (targets.size() != 2 ||
        !quake::isQuantumValueType(targets[0].getType()) ||
        !quake::isQuantumValueType(targets[1].getType())) {
      LLVM_DEBUG(llvm::dbgs() << "operation must have 2 targets\n");
      return failure();
    }

    // Check that these are the same swap op back-to-back.
    auto prev0 = targets[0].template getDefiningOp<quake::SwapOp>();
    if (!prev0) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation 0 must be the same\n");
      return failure();
    }
    auto prev1 = targets[1].template getDefiningOp<quake::SwapOp>();
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
    if (!isa<quake::WireType>(targets[0].getType()) ||
        !isa<quake::WireType>(prevTrgs[0].getType()) ||
        !isa<quake::WireType>(targets[1].getType()) ||
        !isa<quake::WireType>(prevTrgs[1].getType()) ||
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
      if (isa<quake::ControlType>(c.getType()))
        if (!isa<quake::ControlType>(pc.getType()) || c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<quake::WireType>(c.getType()) ||
          !isa<quake::WireType>(pc.getType()) || c != prev0.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

    // Eliminate the back-to-back Hermitian swap gates.
    SmallVector<Value> newOperands;
    newOperands.append(prevCtls.begin(), prevCtls.end());
    newOperands.append(prevTrgs.begin(), prevTrgs.end());
    LLVM_DEBUG(llvm::dbgs() << "eliminated: " << qop << '\n' << prev0 << '\n');
    rewriter.replaceOp(qop, newOperands);
    rewriter.eraseOp(prev0);
    return success();
  }
};

template <typename QOP>
class RotationCombine : public OpRewritePattern<QOP> {
public:
  using Base = OpRewritePattern<QOP>;
  using Base::Base;

  LogicalResult matchAndRewrite(QOP qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    auto targets = qop.getTargets();
    if (targets.size() != 1 ||
        !quake::isQuantumValueType(targets[0].getType())) {
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
    if (!isa<quake::WireType>(trgt.getType()) ||
        !isa<quake::WireType>(prevTrgt.getType()) ||
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
      if (isa<quake::ControlType>(c.getType()))
        if (!isa<quake::ControlType>(pc.getType()) || c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<quake::WireType>(c.getType()) ||
          !isa<quake::WireType>(pc.getType()) || c != prev.getResult(n)) {
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
        p = rewriter.create<arith::NegFOp>(loc, ty, p);
      if (prev.isAdj())
        pp = rewriter.create<arith::NegFOp>(loc, ty, pp);
      newParams.push_back(rewriter.create<arith::AddFOp>(loc, ty, p, pp));
    }

    // Combine the two rotations.
    LLVM_DEBUG(llvm::dbgs() << "combined: " << qop << '\n' << prev << '\n');
    [[maybe_unused]] auto newOp = rewriter.replaceOpWithNewOp<QOP>(
        qop, qop.getResultTypes(), UnitAttr{}, newParams, prevCtls, prevTrgs,
        DenseBoolArrayAttr{});
    rewriter.eraseOp(prev);
    LLVM_DEBUG(llvm::dbgs() << "into: " << newOp << '\n');
    return success();
  }
};

// Z = SS
class DoubleSOp : public OpRewritePattern<quake::SOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::SOp qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    auto targets = qop.getTargets();
    if (targets.size() != 1 ||
        !quake::isQuantumValueType(targets[0].getType())) {
      LLVM_DEBUG(llvm::dbgs() << "operation must have 1 target\n");
      return failure();
    }
    Value trgt = targets[0];

    // Check that these are the same Hermitian op back-to-back.
    auto prev = targets[0].template getDefiningOp<quake::SOp>();
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
    if (!isa<quake::WireType>(trgt.getType()) ||
        !isa<quake::WireType>(prevTrgt.getType()) ||
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
      if (isa<quake::ControlType>(c.getType()))
        if (!isa<quake::ControlType>(pc.getType()) || c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<quake::WireType>(c.getType()) ||
          !isa<quake::WireType>(pc.getType()) || c != prev.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

    // Rewrite the back-to-back S gates.
    LLVM_DEBUG(llvm::dbgs() << "replaced: " << qop << '\n' << prev << '\n');
    rewriter.replaceOpWithNewOp<quake::ZOp>(qop, qop.getResultTypes(),
                                            UnitAttr{}, ValueRange{}, prevCtls,
                                            prevTrgs, DenseBoolArrayAttr{});
    rewriter.eraseOp(prev);
    return success();
  }
};

// S = TT
class DoubleTOp : public OpRewritePattern<quake::TOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::TOp qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    auto targets = qop.getTargets();
    if (targets.size() != 1 ||
        !quake::isQuantumValueType(targets[0].getType())) {
      LLVM_DEBUG(llvm::dbgs() << "operation must have 1 target\n");
      return failure();
    }
    Value trgt = targets[0];

    // Check that these are the same Hermitian op back-to-back.
    auto prev = targets[0].template getDefiningOp<quake::TOp>();
    if (!prev) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must be T\n");
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
    if (!isa<quake::WireType>(trgt.getType()) ||
        !isa<quake::WireType>(prevTrgt.getType()) ||
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
      if (isa<quake::ControlType>(c.getType()))
        if (!isa<quake::ControlType>(pc.getType()) || c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<quake::WireType>(c.getType()) ||
          !isa<quake::WireType>(pc.getType()) || c != prev.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

    // Rewrite the back-to-back S gates.
    LLVM_DEBUG(llvm::dbgs() << "replaced: " << qop << '\n' << prev << '\n');
    rewriter.replaceOpWithNewOp<quake::SOp>(qop, qop.getResultTypes(),
                                            UnitAttr{}, ValueRange{}, prevCtls,
                                            prevTrgs, DenseBoolArrayAttr{});
    rewriter.eraseOp(prev);
    return success();
  }
};

// S = YSX
class ReduceYSX : public OpRewritePattern<quake::XOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::XOp qop,
                                PatternRewriter &rewriter) const override {
    if (qop.getNegatedQubitControls())
      return failure();

    auto targets = qop.getTargets();
    if (targets.size() != 1 ||
        !quake::isQuantumValueType(targets[0].getType())) {
      LLVM_DEBUG(llvm::dbgs() << "operation must have 1 target\n");
      return failure();
    }
    Value trgt = targets[0];

    // Check that these are the same Hermitian op back-to-back.
    auto prev0 = targets[0].template getDefiningOp<quake::SOp>();
    if (!prev0) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must be S\n");
      return failure();
    }
    auto prev = prev0.getTargets()[0].template getDefiningOp<quake::YOp>();
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
    if (!isa<quake::WireType>(trgt.getType()) ||
        !isa<quake::WireType>(prev0Trgt.getType()) ||
        !isa<quake::WireType>(prevTrgt.getType()) ||
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
      if (isa<quake::ControlType>(c.getType()))
        if (!isa<quake::ControlType>(pc.getType()) || c != pc || p0c != pc) {
          LLVM_DEBUG(llvm::dbgs() << "control must be the same\n");
          return failure();
        }
      if (!isa<quake::WireType>(c.getType()) ||
          !isa<quake::WireType>(pc.getType()) ||
          !isa<quake::WireType>(pc.getType()) || c != prev0.getResult(n) ||
          p0c != prev.getResult(n)) {
        LLVM_DEBUG(llvm::dbgs() << "control wire must be threaded\n");
        return failure();
      }
    }

    // Rewrite the back-to-back S gates.
    LLVM_DEBUG(llvm::dbgs() << "replaced: " << qop << '\n'
                            << prev0 << '\n'
                            << prev << '\n');
    rewriter.replaceOpWithNewOp<quake::SOp>(qop, qop.getResultTypes(),
                                            UnitAttr{}, ValueRange{}, prevCtls,
                                            prevTrgs, DenseBoolArrayAttr{});
    rewriter.eraseOp(prev0);
    rewriter.eraseOp(prev);
    return success();
  }
};

namespace {
class QuakeSimplifyPass
    : public cudaq::opt::impl::QuakeSimplifyBase<QuakeSimplifyPass> {
public:
  using QuakeSimplifyBase::QuakeSimplifyBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto *op = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<
        DoubleSOp, DoubleTOp, ReduceYSX, HermitianElimination<quake::HOp>,
        HermitianElimination<quake::SwapOp>, HermitianElimination<quake::XOp>,
        HermitianElimination<quake::YOp>, HermitianElimination<quake::ZOp>,
        RotationCombine<quake::R1Op>, RotationCombine<quake::RxOp>,
        RotationCombine<quake::RyOp>, RotationCombine<quake::RzOp>,
        RotationCombine<quake::PhasedRxOp>>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
