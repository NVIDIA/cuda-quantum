/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "QuakeOperatorUtilities.h"
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_EXPANDCONTROLVEQS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "expand-control-veqs"

using namespace mlir;

namespace {
template <typename OP>
class ExpandPat : public OpRewritePattern<OP> {

public:
  using OpRewritePattern<OP>::OpRewritePattern;

  LogicalResult matchAndRewrite(OP op,
                                PatternRewriter &rewriter) const override {
    auto negatedControls = op.getNegatedQubitControls();
    // Unlike phase lowering, this rewrite cannot preserve an unresolved
    // vector control. Check before materializing any scalar extracts.
    if (cudaq::opt::hasUnresolvedControlVeq(op.getControls()))
      return failure();

    auto expandedControls = cudaq::opt::expandKnownSizedControlVeqs(
        rewriter, op.getLoc(), op.getControls(),
        cudaq::opt::getControlPolarities(op.getControls(), negatedControls));
    if (!expandedControls.didExpand)
      return failure();

    DenseBoolArrayAttr negatedControlsAttr;
    if (negatedControls)
      negatedControlsAttr =
          rewriter.getDenseBoolArrayAttr(expandedControls.polarities);

    // Reconstruct the operation with the new controls
    auto segmentSizes = rewriter.getDenseI32ArrayAttr(
        {static_cast<int32_t>(op.getParameters().size()),
         static_cast<int32_t>(expandedControls.controls.size()),
         static_cast<int32_t>(op.getTargets().size())});

    auto newOp = rewriter.replaceOpWithNewOp<OP>(
        op, op.getResultTypes(), op.getIsAdjAttr(), op.getParameters(),
        expandedControls.controls, op.getTargets(), negatedControlsAttr);

    newOp->setAttr(cudaq::runtime::operandSegmentSizes, segmentSizes);

    return success();
  }
};

struct ExpandControlVeqsPass
    : public cudaq::opt::impl::ExpandControlVeqsBase<ExpandControlVeqsPass> {
  using ExpandControlVeqsBase::ExpandControlVeqsBase;

private:
  template <typename OP>
  static bool checkLegal(OP op) {
    for (auto control : op.getControls()) {
      // Valid ops have no control veqs with a resolvable size (including
      // veq<?> whose size can be determined through RelaxSizeOp).
      if (isa<cudaq::quake::VeqType>(control.getType()))
        if (cudaq::quake::getVeqSize(control))
          return false;
    }

    return true;
  }

public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<
        ExpandPat<cudaq::quake::HOp>, ExpandPat<cudaq::quake::PhasedRxOp>,
        ExpandPat<cudaq::quake::PhaseOp>, ExpandPat<cudaq::quake::R1Op>,
        ExpandPat<cudaq::quake::RxOp>, ExpandPat<cudaq::quake::RyOp>,
        ExpandPat<cudaq::quake::RzOp>, ExpandPat<cudaq::quake::SOp>,
        ExpandPat<cudaq::quake::SwapOp>, ExpandPat<cudaq::quake::TOp>,
        ExpandPat<cudaq::quake::U2Op>, ExpandPat<cudaq::quake::U3Op>,
        ExpandPat<cudaq::quake::XOp>, ExpandPat<cudaq::quake::YOp>,
        ExpandPat<cudaq::quake::ZOp>>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<cudaq::quake::QuakeDialect>();
    target.addDynamicallyLegalOp<cudaq::quake::HOp>(
        checkLegal<cudaq::quake::HOp>);
    target.addDynamicallyLegalOp<cudaq::quake::PhasedRxOp>(
        checkLegal<cudaq::quake::PhasedRxOp>);
    target.addDynamicallyLegalOp<cudaq::quake::PhaseOp>(
        checkLegal<cudaq::quake::PhaseOp>);
    target.addDynamicallyLegalOp<cudaq::quake::R1Op>(
        checkLegal<cudaq::quake::R1Op>);
    target.addDynamicallyLegalOp<cudaq::quake::RxOp>(
        checkLegal<cudaq::quake::RxOp>);
    target.addDynamicallyLegalOp<cudaq::quake::RyOp>(
        checkLegal<cudaq::quake::RyOp>);
    target.addDynamicallyLegalOp<cudaq::quake::RzOp>(
        checkLegal<cudaq::quake::RzOp>);
    target.addDynamicallyLegalOp<cudaq::quake::SOp>(
        checkLegal<cudaq::quake::SOp>);
    target.addDynamicallyLegalOp<cudaq::quake::SwapOp>(
        checkLegal<cudaq::quake::SwapOp>);
    target.addDynamicallyLegalOp<cudaq::quake::TOp>(
        checkLegal<cudaq::quake::TOp>);
    target.addDynamicallyLegalOp<cudaq::quake::U2Op>(
        checkLegal<cudaq::quake::U2Op>);
    target.addDynamicallyLegalOp<cudaq::quake::U3Op>(
        checkLegal<cudaq::quake::U3Op>);
    target.addDynamicallyLegalOp<cudaq::quake::XOp>(
        checkLegal<cudaq::quake::XOp>);
    target.addDynamicallyLegalOp<cudaq::quake::YOp>(
        checkLegal<cudaq::quake::YOp>);
    target.addDynamicallyLegalOp<cudaq::quake::ZOp>(
        checkLegal<cudaq::quake::ZOp>);
    if (failed(applyPartialConversion(func.getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
