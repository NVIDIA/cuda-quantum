/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_APPLYCONTROLNEGATIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

/// Replace any operations with negative controls with the same
/// operation with negative controls and the addition of X operations
/// on each control qubit before and after the operation.
template <typename Op>
class ReplaceNegativeControl : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto negations = op.getNegatedQubitControls();
    if (!negations.has_value())
      return failure();

    for (auto negationIter : llvm::enumerate(negations.value()))
      if (negationIter.value())
        rewriter.create<quake::XOp>(
            loc, ValueRange(),
            ValueRange{op.getControls()[negationIter.index()]});

    if constexpr (std::is_same_v<Op, quake::ExpPauliOp>) {
      rewriter.create<quake::ExpPauliOp>(
          loc, TypeRange{}, op.getIsAdjAttr(), op.getParameters(),
          op.getControls(), op.getTargets(), op.getNegatedQubitControlsAttr(),
          op.getPauli(), op.getPauliLiteralAttr());
    } else if constexpr (std::is_same_v<Op, quake::CustomUnitarySymbolOp>) {
      rewriter.create<Op>(loc, op.getGeneratorAttr(), op.getIsAdj(),
                          op.getParameters(), op.getControls(),
                          op.getTargets());
    } else {
      rewriter.create<Op>(loc, op.getIsAdj(), op.getParameters(),
                          op.getControls(), op.getTargets());
    }

    for (auto negationIter : llvm::enumerate(negations.value()))
      if (negationIter.value())
        rewriter.create<quake::XOp>(
            loc, ValueRange(),
            ValueRange{op.getControls()[negationIter.index()]});
    rewriter.eraseOp(op);

    return success();
  }
};

namespace {

struct ApplyControlNegationsPass
    : public cudaq::opt::impl::ApplyControlNegationsBase<
          ApplyControlNegationsPass> {
  using ApplyControlNegationsBase::ApplyControlNegationsBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<
        ReplaceNegativeControl<quake::XOp>, ReplaceNegativeControl<quake::YOp>,
        ReplaceNegativeControl<quake::ZOp>, ReplaceNegativeControl<quake::HOp>,
        ReplaceNegativeControl<quake::SOp>, ReplaceNegativeControl<quake::TOp>,
        ReplaceNegativeControl<quake::RxOp>,
        ReplaceNegativeControl<quake::RyOp>,
        ReplaceNegativeControl<quake::RzOp>,
        ReplaceNegativeControl<quake::R1Op>,
        ReplaceNegativeControl<quake::U3Op>,
        ReplaceNegativeControl<quake::SwapOp>,
        ReplaceNegativeControl<quake::ExpPauliOp>,
        ReplaceNegativeControl<quake::CustomUnitarySymbolOp>>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<cudaq::cc::CCDialect, arith::ArithDialect,
                           LLVM::LLVMDialect>();
    target.addDynamicallyLegalDialect<quake::QuakeDialect>([](Operation *op) {
      auto quantumOp = dyn_cast<quake::OperatorInterface>(op);
      if (!quantumOp)
        return true;

      auto negations = quantumOp.getNegatedControls();
      if (!negations.has_value())
        return true;

      for (auto negation : negations.value())
        if (negation)
          return false;

      return true;
    });
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      funcOp->emitOpError("could not replace negations");
      signalPassFailure();
    }
  }
};
} // namespace
