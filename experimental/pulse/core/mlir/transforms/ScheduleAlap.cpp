// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// ALAP scheduling pass for the Pulse dialect.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
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

static int64_t getWaveformDuration(mlir::Value wfVal) {
  auto *defOp = wfVal.getDefiningOp();
  if (!defOp)
    return 0;
  mlir::Value durVal;
  if (auto op = mlir::dyn_cast<pulse::GaussianPulseOp>(defOp))
    durVal = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::SquarePulseOp>(defOp))
    durVal = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::DRAGPulseOp>(defOp))
    durVal = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::CosinePulseOp>(defOp))
    durVal = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::TanhRampOp>(defOp))
    durVal = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::GaussianSquarePulseOp>(defOp))
    durVal = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::CustomOp>(defOp))
    durVal = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::CustomSamplesOp>(defOp))
    return static_cast<int64_t>(op.getSamples().size());
  else
    return 0;

  return traceConstantI64(durVal).value_or(0);
}

struct PulseScheduleAlapPass
    : public mlir::PassWrapper<PulseScheduleAlapPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseScheduleAlapPass)

  llvm::StringRef getArgument() const override { return "pulse-schedule-alap"; }
  llvm::StringRef getDescription() const override {
    return "Assign start_vtu/duration_vtu via ALAP scheduling";
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    llvm::DenseMap<mlir::Value, int64_t> lineTime;
    auto i64Ty = mlir::IntegerType::get(&getContext(), 64);

    funcOp.walk([&](mlir::Operation *op) {
      if (auto driveOp = mlir::dyn_cast<pulse::DriveOp>(op)) {
        auto line = driveOp.getLine();
        int64_t t = lineTime.lookup(line);
        int64_t dur = 0;
        if (auto a = driveOp->getAttrOfType<mlir::IntegerAttr>("duration_vtu"))
          dur = a.getInt();
        else
          dur = getWaveformDuration(driveOp.getPulse());

        driveOp->setAttr("start_vtu", mlir::IntegerAttr::get(i64Ty, t));
        driveOp->setAttr("duration_vtu", mlir::IntegerAttr::get(i64Ty, dur));
        lineTime[driveOp.getUpdatedLine()] = t + dur;

      } else if (auto waitOp = mlir::dyn_cast<pulse::WaitOp>(op)) {
        auto line = waitOp.getLine();
        int64_t t = lineTime.lookup(line);
        int64_t dur = 0;
        if (auto durIntOp = waitOp.getDuration()
                                .getDefiningOp<pulse::DurationFromIntOp>()) {
          if (auto cst = traceConstantI64(durIntOp.getCycles()))
            dur = *cst;
        }
        waitOp->setAttr("start_vtu", mlir::IntegerAttr::get(i64Ty, t));
        waitOp->setAttr("duration_vtu", mlir::IntegerAttr::get(i64Ty, dur));
        lineTime[waitOp.getResult()] = t + dur;

      } else if (auto syncOp = mlir::dyn_cast<pulse::SyncOp>(op)) {
        int64_t maxT = 0;
        for (auto line : syncOp.getLines()) {
          int64_t t = lineTime.lookup(line);
          if (t > maxT)
            maxT = t;
        }
        for (auto result : syncOp.getResults()) {
          lineTime[result] = maxT;
        }
      }
    });
  }
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseScheduleAlapPass() {
  return std::make_unique<PulseScheduleAlapPass>();
}
} // namespace pulse
