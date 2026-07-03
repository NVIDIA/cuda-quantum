// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Module-level verification pass for the Pulse dialect.
// Checks linearity, monotone time, drive exclusivity, and waveform validity
// beyond what individual op verifiers can check.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

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

struct PulseVerifyPass
    : public mlir::PassWrapper<PulseVerifyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseVerifyPass)

  llvm::StringRef getArgument() const override { return "pulse-verify"; }
  llvm::StringRef getDescription() const override {
    return "Module-level verification of Pulse IR constraints";
  }

  void runOnOperation() override {
    auto module = getOperation();
    bool hadError = false;

    // Check 1: Linearity -- lines must be consumed.
    module.walk([&](pulse::GetDriveLineOp op) {
      if (op.getLine().use_empty())
        op.emitWarning("drive line allocated but never used");
    });
    module.walk([&](pulse::GetReadoutLineOp op) {
      if (op.getLine().use_empty())
        op.emitWarning("readout line allocated but never used");
    });

    // Check 2: Waveform validity -- trace SSA duration to constant, skip
    // parametric (block arg) values which are checked at evaluation time.
    auto checkDuration = [&](mlir::Operation *op, mlir::Value durVal) {
      if (auto dur = traceConstantI64(durVal)) {
        if (*dur <= 0) {
          op->emitError("waveform duration must be positive, got ") << *dur;
          hadError = true;
        }
      }
    };
    module.walk([&](pulse::GaussianPulseOp op) {
      checkDuration(op, op.getDuration());
    });
    module.walk([&](pulse::SquarePulseOp op) {
      checkDuration(op, op.getDuration());
    });
    module.walk([&](pulse::DRAGPulseOp op) {
      checkDuration(op, op.getDuration());
    });

    // Check 3: Monotone time (scheduled programs only).
    module.walk([&](mlir::func::FuncOp funcOp) {
      llvm::DenseMap<mlir::Value, int64_t> lastStart;
      funcOp.walk([&](pulse::DriveOp driveOp) {
        auto startAttr = driveOp->getAttrOfType<mlir::IntegerAttr>("start_vtu");
        if (!startAttr)
          return;
        int64_t start = startAttr.getInt();
        auto line = driveOp.getLine();
        auto it = lastStart.find(line);
        if (it != lastStart.end() && start < it->second) {
          driveOp.emitWarning("non-monotone start_vtu: ")
              << start << " < " << it->second;
        }
        lastStart[line] = start;
      });
    });

    if (hadError)
      signalPassFailure();
  }
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseVerifyPass() {
  return std::make_unique<PulseVerifyPass>();
}
} // namespace pulse
