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
    // Unfortunately, we may grow the # of controls as a result of this rewrite,
    // so we must track a new list of controls and reconstruct the operands
    // instead of replacing the controls in place.
    SmallVector<Value> newControls;
    bool update = false;

    // Search through the controls for veqs with known sizes
    for (auto [index, control] : llvm::enumerate(op.getControls())) {
      if (auto veq = dyn_cast<quake::VeqType>(control.getType())) {
        if (!veq.hasSpecifiedSize())
          return failure();

        // For each of the qubits in the veq, create an extraction instruction
        // The result of the extraction will be a new control
        // The veq is not added the newControls, so it will be dropped
        for (size_t i = 0; i < veq.getSize(); ++i) {
          auto ext =
              rewriter.create<quake::ExtractRefOp>(op.getLoc(), control, i);
          newControls.push_back(ext);
          update = true;
        }
      } else {
        newControls.push_back(control);
      }
    }

    if (!update)
      return failure();

    // Reconstruct the operation with the new controls
    auto segmentSizes = rewriter.getDenseI32ArrayAttr(
        {static_cast<int32_t>(op.getParameters().size()),
         static_cast<int32_t>(newControls.size()),
         static_cast<int32_t>(op.getTargets().size())});

    auto newOp = rewriter.replaceOpWithNewOp<OP>(
        op, op.getIsAdj(), op.getParameters(), newControls, op.getTargets(),
        op.getNegatedQubitControlsAttr());

    newOp->setAttr("operand_segment_sizes", segmentSizes);

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
      // Valid ops have no control veqs with a known size
      if (auto veq = dyn_cast<quake::VeqType>(control.getType()))
        if (veq.hasSpecifiedSize())
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
        ExpandPat<quake::HOp>, ExpandPat<quake::PhasedRxOp>,
        ExpandPat<quake::R1Op>, ExpandPat<quake::RxOp>, ExpandPat<quake::RyOp>,
        ExpandPat<quake::RzOp>, ExpandPat<quake::SOp>, ExpandPat<quake::SwapOp>,
        ExpandPat<quake::TOp>, ExpandPat<quake::U2Op>, ExpandPat<quake::U3Op>,
        ExpandPat<quake::XOp>, ExpandPat<quake::YOp>, ExpandPat<quake::ZOp>>(
        ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addDynamicallyLegalOp<quake::HOp>(checkLegal<quake::HOp>);
    target.addDynamicallyLegalOp<quake::PhasedRxOp>(
        checkLegal<quake::PhasedRxOp>);
    target.addDynamicallyLegalOp<quake::R1Op>(checkLegal<quake::R1Op>);
    target.addDynamicallyLegalOp<quake::RxOp>(checkLegal<quake::RxOp>);
    target.addDynamicallyLegalOp<quake::RyOp>(checkLegal<quake::RyOp>);
    target.addDynamicallyLegalOp<quake::RzOp>(checkLegal<quake::RzOp>);
    target.addDynamicallyLegalOp<quake::SOp>(checkLegal<quake::SOp>);
    target.addDynamicallyLegalOp<quake::SwapOp>(checkLegal<quake::SwapOp>);
    target.addDynamicallyLegalOp<quake::TOp>(checkLegal<quake::TOp>);
    target.addDynamicallyLegalOp<quake::U2Op>(checkLegal<quake::U2Op>);
    target.addDynamicallyLegalOp<quake::U3Op>(checkLegal<quake::U3Op>);
    target.addDynamicallyLegalOp<quake::XOp>(checkLegal<quake::XOp>);
    target.addDynamicallyLegalOp<quake::YOp>(checkLegal<quake::YOp>);
    target.addDynamicallyLegalOp<quake::ZOp>(checkLegal<quake::ZOp>);
    if (failed(applyPartialConversion(func.getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
