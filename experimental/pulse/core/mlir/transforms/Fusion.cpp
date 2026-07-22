// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fusion pass: merge adjacent drive ops on the same line with same-amplitude
// square waveforms into a single drive with summed duration.

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

static std::optional<int64_t> traceConstantI64(mlir::Value v) {
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static bool sameSSAValue(mlir::Value a, mlir::Value b) {
  if (a == b)
    return true;
  auto *aOp = a.getDefiningOp();
  auto *bOp = b.getDefiningOp();
  if (!aOp || !bOp)
    return false;
  auto aCst = mlir::dyn_cast<mlir::arith::ConstantFloatOp>(aOp);
  auto bCst = mlir::dyn_cast<mlir::arith::ConstantFloatOp>(bOp);
  if (aCst && bCst)
    return aCst.value().bitwiseIsEqual(bCst.value());
  return false;
}

struct FuseAdjacentSquareDrives
    : public mlir::OpRewritePattern<pulse::DriveOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::DriveOp firstDrive,
                  mlir::PatternRewriter &rewriter) const override {
    auto firstWf = firstDrive.getPulse().getDefiningOp<pulse::SquarePulseOp>();
    if (!firstWf)
      return mlir::failure();

    auto updatedLine = firstDrive.getUpdatedLine();
    if (!updatedLine.hasOneUse())
      return mlir::failure();

    auto *user = *updatedLine.getUsers().begin();
    auto secondDrive = mlir::dyn_cast<pulse::DriveOp>(user);
    if (!secondDrive || secondDrive.getLine() != updatedLine)
      return mlir::failure();

    auto secondWf =
        secondDrive.getPulse().getDefiningOp<pulse::SquarePulseOp>();
    if (!secondWf)
      return mlir::failure();

    if (!sameSSAValue(firstWf.getAmpReal(), secondWf.getAmpReal()) ||
        !sameSSAValue(firstWf.getAmpImag(), secondWf.getAmpImag()))
      return mlir::failure();

    if (secondDrive.getTone() != firstDrive.getUpdatedTone())
      return mlir::failure();

    auto dur1 = traceConstantI64(firstWf.getDuration());
    auto dur2 = traceConstantI64(secondWf.getDuration());
    if (!dur1 || !dur2)
      return mlir::failure();
    int64_t fusedDur = *dur1 + *dur2;

    auto loc = firstDrive.getLoc();
    auto i64Ty = rewriter.getIntegerType(64);
    auto durConst =
        rewriter.create<mlir::arith::ConstantIntOp>(loc, fusedDur, 64);
    auto fusedWf = rewriter.create<pulse::SquarePulseOp>(
        loc, firstWf.getType(), durConst.getResult(), firstWf.getAmpReal(),
        firstWf.getAmpImag());

    auto fusedDrive = rewriter.create<pulse::DriveOp>(
        loc, firstDrive.getUpdatedLine().getType(),
        firstDrive.getUpdatedTone().getType(), firstDrive.getLine(),
        fusedWf.getResult(), firstDrive.getTone());

    if (auto a = firstDrive->getAttrOfType<mlir::IntegerAttr>("start_vtu"))
      fusedDrive->setAttr("start_vtu", a);
    fusedDrive->setAttr("duration_vtu", rewriter.getI64IntegerAttr(fusedDur));
    fusedDrive->setAttr("fused", rewriter.getUnitAttr());

    rewriter.replaceOp(secondDrive, fusedDrive->getResults());
    rewriter.replaceOp(firstDrive, fusedDrive->getResults());
    return mlir::success();
  }
};

struct PulseFusionPass
    : public mlir::PassWrapper<PulseFusionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseFusionPass)

  llvm::StringRef getArgument() const override { return "pulse-fusion"; }
  llvm::StringRef getDescription() const override {
    return "Fuse adjacent same-amplitude square-pulse drives into one";
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FuseAdjacentSquareDrives>(&getContext());
    mlir::GreedyRewriteConfig config;
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                 std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseFusionPass() {
  return std::make_unique<PulseFusionPass>();
}
} // namespace pulse
