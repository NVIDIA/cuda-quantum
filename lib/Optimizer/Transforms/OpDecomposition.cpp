/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXDialect.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXInterfaces.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct CliffordT : public ConversionTarget {
  CliffordT(MLIRContext &context) : ConversionTarget(context) {
    addDynamicallyLegalDialect<qtx::QTXDialect>([](Operation *op) {
      if (auto tOp = dyn_cast<qtx::TOp>(op))
        return tOp.getControls().size() == 0;
      if (auto optor = dyn_cast<qtx::OperatorInterface>(op))
        return optor.isClifford();
      return true; // In decomposition, non-quantum operators are all legal.
    });
  }
};

template <typename Op>
Value createOp(PatternRewriter &rewriter, Location loc, Value &target) {
  return rewriter.create<Op>(loc, ValueRange(), target).getResult(0);
}

template <typename Op>
Value createOp(PatternRewriter &rewriter, Location loc, bool isAdj,
               Value &target) {
  return rewriter.create<Op>(loc, isAdj, target).getResult(0);
}

template <typename Op>
Value createOp(PatternRewriter &rewriter, Location loc, ValueRange controls,
               Value &target) {
  return rewriter.create<Op>(loc, controls, target).getResult(0);
}

//===----------------------------------------------------------------------===//
// ZOp decompositions
//===----------------------------------------------------------------------===//

// Two-control decomposition
//                                                                  ┌───┐
//  ───●────  ──────────────●───────────────────●──────●─────────●──┤ T ├
//     │                    │                   │      │         │  └───┘
//     │                    │                   │    ┌─┴─┐┌───┐┌─┴─┐┌───┐
//  ───●─── = ────●─────────┼─────────●─────────┼────┤ X ├┤ ┴ ├┤ X ├┤ T ├
//     │          │         │         │         │    └───┘└───┘└───┘└───┘
//   ┌─┴─┐      ┌─┴─┐┌───┐┌─┴─┐┌───┐┌─┴─┐┌───┐┌─┴─┐                 ┌───┐
//  ─┤ z ├─   ──┤ X ├┤ ┴ ├┤ X ├┤ T ├┤ X ├┤ ┴ ├┤ X ├─────────────────┤ T ├
//   └───┘      └───┘└───┘└───┘└───┘└───┘└───┘└───┘                 └───┘
//
// NOTE: `┴` denotes the adjoint of `qtx.t`.
struct ZOpDecomposition : public OpRewritePattern<qtx::ZOp> {
  using OpRewritePattern<qtx::ZOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(qtx::ZOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getControls().size() != 2)
      return failure();

    Location loc = op->getLoc();
    Value c[2] = {op.getControls()[0], op.getControls()[1]};
    Value t = op.getTarget();

    t = createOp<qtx::XOp>(rewriter, loc, c[1], t);
    t = createOp<qtx::TOp>(rewriter, loc, /*isAdj=*/true, t);
    t = createOp<qtx::XOp>(rewriter, loc, c[0], t);
    t = createOp<qtx::TOp>(rewriter, loc, t);
    t = createOp<qtx::XOp>(rewriter, loc, c[1], t);
    t = createOp<qtx::TOp>(rewriter, loc, /*isAdj=*/true, t);
    t = createOp<qtx::XOp>(rewriter, loc, c[0], t);
    t = createOp<qtx::TOp>(rewriter, loc, t);

    c[1] = createOp<qtx::XOp>(rewriter, loc, c[0], c[1]);
    c[1] = createOp<qtx::TOp>(rewriter, loc, /*isAdj=*/true, c[1]);
    c[1] = createOp<qtx::XOp>(rewriter, loc, c[0], c[1]);
    c[1] = createOp<qtx::TOp>(rewriter, loc, c[1]);

    c[0] = createOp<qtx::TOp>(rewriter, loc, c[0]);

    op.getResult(0).replaceAllUsesWith(t);
    op.getControls()[0].replaceUsesWithIf(c[0], [&](OpOperand &use) -> bool {
      return !use.getOwner()->isBeforeInBlock(op);
    });
    op.getControls()[1].replaceUsesWithIf(c[1], [&](OpOperand &use) -> bool {
      return !use.getOwner()->isBeforeInBlock(op);
    });
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// XOp decompositions
//===----------------------------------------------------------------------===//

// Two-control decomposition
//
//  ───●────  ────────●────────
//     │              │
//     │              │
//  ───●─── = ────────●────────
//     │              │
//   ┌─┴─┐     ┌───┐┌─┴─┐┌───┐
//  ─┤ x ├─   ─┤ H ├┤ Z ├┤ H ├─
//   └───┘     └───┘└───┘└───┘
struct XOpDecomposition : public OpRewritePattern<qtx::XOp> {
  using OpRewritePattern<qtx::XOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(qtx::XOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getControls().size() != 2)
      return failure();

    Location loc = op->getLoc();
    Value t = op.getTarget();
    t = createOp<qtx::HOp>(rewriter, loc, t);
    t = createOp<qtx::ZOp>(rewriter, loc, op.getControls(), t);
    t = createOp<qtx::HOp>(rewriter, loc, t);
    op.getResult(0).replaceAllUsesWith(t);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

struct OpDecomposition
    : public cudaq::opt::OpDecompositionBase<OpDecomposition> {
  void runOnOperation() override {
    auto circuit = cast<qtx::CircuitOp>(getOperation());
    MLIRContext *context = circuit.getContext();
    RewritePatternSet patterns(context);
    patterns.insert<XOpDecomposition, ZOpDecomposition>(context);
    CliffordT target(*context);
    if (failed(applyPartialConversion(circuit, target, std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> cudaq::opt::createOpDecompositionPass() {
  return std::make_unique<OpDecomposition>();
}
