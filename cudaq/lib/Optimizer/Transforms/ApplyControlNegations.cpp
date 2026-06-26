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
      // The X conjugation above already realizes the negated controls, so the
      // recreated op must NOT carry the negation attribute - otherwise it stays
      // illegal and the partial conversion cannot converge.
      cudaq::quake::ExpPauliOp::create(
          rewriter, loc, TypeRange{}, op.getIsAdjAttr(), op.getParameters(),
          op.getControls(), op.getTargets(), mlir::DenseBoolArrayAttr{},
          op.getPauli(), op.getPauliLiteralAttr());
    } else if constexpr (std::is_same_v<Op,
                                        cudaq::quake::CustomUnitaryCallOp>) {
      // The X conjugation above realizes the negated controls, so the recreated
      // op uses plain (positive) controls and carries no negation attribute.
      cudaq::quake::CustomUnitaryCallOp::create(
          rewriter, loc, op.getGeneratorAttr(), op.getIsAdj(),
          op.getParameters(), op.getControls(), op.getTargets());
    } else if constexpr (std::is_same_v<
                             Op, cudaq::quake::CustomUnitaryConstantOp>) {
      // The X conjugation above realizes the negated controls, so the recreated
      // op uses plain (positive) controls and carries no negation attribute.
      cudaq::quake::CustomUnitaryConstantOp::create(
          rewriter, loc, op.getMatrixAttr(), op.getIsAdj(), op.getParameters(),
          op.getControls(), op.getTargets(), {});
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
                ReplaceNegativeControl<cudaq::quake::CustomUnitaryCallOp>,
                ReplaceNegativeControl<cudaq::quake::CustomUnitaryConstantOp>>(
            ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<cudaq::cc::CCDialect, arith::ArithDialect,
                           LLVM::LLVMDialect>();
    const bool preserveNegatedControls = preserveGateControlPolarity;
    target.addDynamicallyLegalDialect<cudaq::quake::QuakeDialect>(
        [preserveNegatedControls](Operation *op) {
          auto quantumOp = dyn_cast<cudaq::quake::OperatorInterface>(op);
          if (!quantumOp)
            return true;

          auto negations = quantumOp.getNegatedControls();
          if (!negations.has_value())
            return true;

          bool anyNegated = false;
          for (auto negation : negations.value())
            anyNegated |= negation;
          if (!anyNegated)
            return true;

          // Preserve negations only for gates enabled on the native-control
          // runtime path. Everything else - exp_pauli, custom unitary, and
          // built-ins kept on the fallback path (phased_rx, u2, u3, swap), is
          // still expanded into X conjugation.
          if (preserveNegatedControls) {
            if (isa<cudaq::quake::PhasedRxOp, cudaq::quake::U2Op,
                    cudaq::quake::U3Op, cudaq::quake::SwapOp,
                    cudaq::quake::ExpPauliOp, cudaq::quake::CustomUnitaryCallOp,
                    cudaq::quake::CustomUnitaryConstantOp>(op))
              return false;
            // All other quake ops carrying negations at this point are
            // preservable built-ins.
            return true;
          }

          return false;
        });
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      funcOp->emitOpError("could not replace negations");
      signalPassFailure();
    }
  }
};
} // namespace
