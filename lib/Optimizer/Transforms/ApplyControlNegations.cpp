/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
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
        cudaq::quake::XOp::create(
            rewriter, loc, ValueRange(),
            ValueRange{op.getControls()[negationIter.index()]});

    if constexpr (std::is_same_v<Op, cudaq::quake::ExpPauliOp>) {
      cudaq::quake::ExpPauliOp::create(
          rewriter, loc, TypeRange{}, op.getIsAdjAttr(), op.getParameters(),
          op.getControls(), op.getTargets(), op.getNegatedQubitControlsAttr(),
          op.getPauli(), op.getPauliLiteralAttr());
    } else if constexpr (std::is_same_v<Op,
                                        cudaq::quake::CustomUnitarySymbolOp>) {
      Op::create(rewriter, loc, op.getGeneratorAttr(), op.getIsAdj(),
                 op.getParameters(), op.getControls(), op.getTargets());
    } else {
      Op::create(rewriter, loc, op.getIsAdj(), op.getParameters(),
                 op.getControls(), op.getTargets());
    }

    for (auto negationIter : llvm::enumerate(negations.value()))
      if (negationIter.value())
        cudaq::quake::XOp::create(
            rewriter, loc, ValueRange(),
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
    patterns
        .insert<ReplaceNegativeControl<cudaq::quake::XOp>,
                ReplaceNegativeControl<cudaq::quake::YOp>,
                ReplaceNegativeControl<cudaq::quake::ZOp>,
                ReplaceNegativeControl<cudaq::quake::HOp>,
                ReplaceNegativeControl<cudaq::quake::SOp>,
                ReplaceNegativeControl<cudaq::quake::TOp>,
                ReplaceNegativeControl<cudaq::quake::RxOp>,
                ReplaceNegativeControl<cudaq::quake::RyOp>,
                ReplaceNegativeControl<cudaq::quake::RzOp>,
                ReplaceNegativeControl<cudaq::quake::R1Op>,
                ReplaceNegativeControl<cudaq::quake::U3Op>,
                ReplaceNegativeControl<cudaq::quake::SwapOp>,
                ReplaceNegativeControl<cudaq::quake::ExpPauliOp>,
                ReplaceNegativeControl<cudaq::quake::CustomUnitarySymbolOp>>(
            ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<cudaq::cc::CCDialect, arith::ArithDialect,
                           LLVM::LLVMDialect>();
    target.addDynamicallyLegalDialect<cudaq::quake::QuakeDialect>(
        [](Operation *op) {
          auto quantumOp = dyn_cast<cudaq::quake::OperatorInterface>(op);
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
