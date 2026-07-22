// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Pulse canonicalization pass: redundant sync elimination, dead line
// elimination, waveform CSE.

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

/// Remove sync ops with only one operand (no-op synchronization).
struct RemoveSingleOperandSync : public mlir::OpRewritePattern<pulse::SyncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::SyncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getNumOperands() <= 1) {
      if (op.getNumResults() == 1 && op.getNumOperands() == 1) {
        rewriter.replaceOp(op, op.getOperand(0));
      } else if (op.getNumResults() == 0) {
        rewriter.eraseOp(op);
      } else {
        return mlir::failure();
      }
      return mlir::success();
    }
    return mlir::failure();
  }
};

/// Remove consecutive duplicate sync ops on the same set of lines.
struct RemoveRedundantSync : public mlir::OpRewritePattern<pulse::SyncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(pulse::SyncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto prevOp = op->getPrevNode();
    if (!prevOp)
      return mlir::failure();

    auto prevSync = mlir::dyn_cast<pulse::SyncOp>(prevOp);
    if (!prevSync)
      return mlir::failure();

    if (prevSync.getNumResults() != op.getNumOperands())
      return mlir::failure();

    bool allMatch = true;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (op.getOperand(i) != prevSync.getResult(i)) {
        allMatch = false;
        break;
      }
    }

    if (!allMatch)
      return mlir::failure();

    // This sync immediately follows the previous one on the same lines
    rewriter.replaceOp(op, prevSync.getResults());
    return mlir::success();
  }
};

struct PulseCanonicalizePass
    : public mlir::PassWrapper<PulseCanonicalizePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseCanonicalizePass)

  llvm::StringRef getArgument() const override { return "pulse-canonicalize"; }
  llvm::StringRef getDescription() const override {
    return "Pulse-level canonicalization: sync elimination, waveform CSE";
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RemoveSingleOperandSync>(&getContext());
    patterns.add<RemoveRedundantSync>(&getContext());

    mlir::GreedyRewriteConfig config;
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                 std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseCanonicalizePass() {
  return std::make_unique<PulseCanonicalizePass>();
}
} // namespace pulse
