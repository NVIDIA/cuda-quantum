// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Virtual-Z pass: fold shift_phase ops into the next drive's waveform phase.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

namespace {

struct FoldShiftPhaseIntoDrive
    : public mlir::OpRewritePattern<pulse::ShiftPhaseOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::ShiftPhaseOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto result = op.getResult();
    if (!result.hasOneUse())
      return mlir::failure();

    auto *user = *result.getUsers().begin();
    auto driveOp = mlir::dyn_cast<pulse::DriveOp>(user);
    if (!driveOp || driveOp.getTone() != result)
      return mlir::failure();

    double existing = 0.0;
    if (auto attr = driveOp->getAttrOfType<mlir::FloatAttr>("phase_offset"))
      existing = attr.getValueAsDouble();

    auto phaseVal = op.getPhaseRad();
    if (auto cst = phaseVal.getDefiningOp<mlir::arith::ConstantFloatOp>()) {
      double delta = cst.value().convertToDouble();
      driveOp->setAttr("phase_offset",
                       rewriter.getF64FloatAttr(existing + delta));
    } else {
      return mlir::failure();
    }

    // Drive now consumes the original tone (input to shift_phase)
    driveOp.getToneMutable().assign(op.getTone());
    rewriter.replaceOp(op, op.getTone());
    return mlir::success();
  }
};

struct VirtualZPass
    : public mlir::PassWrapper<VirtualZPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VirtualZPass)

  llvm::StringRef getArgument() const override { return "pulse-virtual-z"; }
  llvm::StringRef getDescription() const override {
    return "Fold shift_phase ops into adjacent drive ops as phase attributes";
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FoldShiftPhaseIntoDrive>(&getContext());
    mlir::GreedyRewriteConfig config;
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                 std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createVirtualZPass() {
  return std::make_unique<VirtualZPass>();
}
} // namespace pulse
