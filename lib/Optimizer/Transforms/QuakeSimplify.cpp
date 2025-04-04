/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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

    // Check that these are the same rotation op back-to-back.
    auto prev = targets[0].template getDefiningOp<QOP>();
    if (!prev) {
      LLVM_DEBUG(llvm::dbgs() << "previous operation must be the same\n");
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

    auto params = qop.getParameters();
    auto prevParams = prev.getParameters();
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
        HermitianElimination<quake::HOp>, HermitianElimination<quake::SwapOp>,
        HermitianElimination<quake::XOp>, HermitianElimination<quake::YOp>,
        HermitianElimination<quake::ZOp>, RotationCombine<quake::R1Op>,
        RotationCombine<quake::RxOp>, RotationCombine<quake::RyOp>,
        RotationCombine<quake::RzOp>, RotationCombine<quake::U2Op>,
        RotationCombine<quake::U3Op>, RotationCombine<quake::PhasedRxOp>>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
