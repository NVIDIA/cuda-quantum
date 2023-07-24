/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct R1ToRz : public OpRewritePattern<quake::R1Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(quake::R1Op r1Op,
                                PatternRewriter &rewriter) const override {
    if (!r1Op.getControls().empty())
      return failure();

    rewriter.replaceOpWithNewOp<quake::RzOp>(
        r1Op, r1Op.getParameters(), r1Op.getControls(), r1Op.getTargets());
    return success();
  }
};

class QuantinuumNaiveR1ToRzPass
    : public cudaq::opt::QuantinuumNaiveR1ToRzBase<QuantinuumNaiveR1ToRzPass> {
public:
  using QuantinuumNaiveR1ToRzBase::QuantinuumNaiveR1ToRzBase;

  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<R1ToRz>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      op->emitOpError("could not lower r1 to rz for quantinuum.");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createQuantinuumNaiveR1ToRz() {
  return std::make_unique<QuantinuumNaiveR1ToRzPass>();
}
