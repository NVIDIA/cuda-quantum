/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Deterministic ASAP and ALAP scheduling passes for the Pulse dialect.

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

namespace {

static std::optional<int64_t> traceConstantI64(mlir::Value value) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantIntOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return attr.getInt();
  return std::nullopt;
}

static int64_t getWaveformDuration(mlir::Value waveform) {
  auto *definition = waveform.getDefiningOp();
  if (!definition)
    return 0;

  mlir::Value duration;
  if (auto op = mlir::dyn_cast<pulse::GaussianPulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::SquarePulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::DRAGPulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::CosinePulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::TanhRampOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::GaussianSquarePulseOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::CustomOp>(definition))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::CustomSamplesOp>(definition))
    return static_cast<int64_t>(op.getSamples().size());
  else if (auto op = mlir::dyn_cast<pulse::PulseAddOp>(definition))
    return getWaveformDuration(op.getLhs());
  else if (auto op = mlir::dyn_cast<pulse::PulseSubOp>(definition))
    return getWaveformDuration(op.getLhs());
  else if (auto op = mlir::dyn_cast<pulse::PulseMulOp>(definition))
    return getWaveformDuration(op.getLhs());
  else if (auto op = mlir::dyn_cast<pulse::PulseScaleOp>(definition))
    return getWaveformDuration(op.getPulse());
  else if (auto op = mlir::dyn_cast<pulse::PulseNegOp>(definition))
    return getWaveformDuration(op.getPulse());
  else
    return 0;

  return traceConstantI64(duration).value_or(0);
}

static int64_t getWaitDuration(pulse::WaitOp wait) {
  if (auto conversion =
          wait.getDuration().getDefiningOp<pulse::DurationFromIntOp>())
    return traceConstantI64(conversion.getCycles()).value_or(0);
  return 0;
}

static void setTiming(mlir::Operation *operation, mlir::IntegerType i64Type,
                      int64_t start, int64_t duration) {
  operation->setAttr("start_vtu", mlir::IntegerAttr::get(i64Type, start));
  operation->setAttr("duration_vtu", mlir::IntegerAttr::get(i64Type, duration));
}

/// Assign an ASAP schedule and return its makespan.
static int64_t scheduleAsap(mlir::func::FuncOp function) {
  llvm::DenseMap<mlir::Value, int64_t> lineReady;
  auto i64Type = mlir::IntegerType::get(function.getContext(), 64);
  int64_t makespan = 0;

  function.walk([&](mlir::Operation *operation) {
    if (auto drive = mlir::dyn_cast<pulse::DriveOp>(operation)) {
      int64_t start = lineReady.lookup(drive.getLine());
      int64_t duration = getWaveformDuration(drive.getPulse());
      setTiming(operation, i64Type, start, duration);
      lineReady[drive.getUpdatedLine()] = start + duration;
      makespan = std::max(makespan, start + duration);
    } else if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(operation)) {
      int64_t start = lineReady.lookup(readout.getLine());
      int64_t duration = getWaveformDuration(readout.getPulse());
      setTiming(operation, i64Type, start, duration);
      lineReady[readout.getUpdatedLine()] = start + duration;
      makespan = std::max(makespan, start + duration);
    } else if (auto wait = mlir::dyn_cast<pulse::WaitOp>(operation)) {
      int64_t start = lineReady.lookup(wait.getLine());
      int64_t duration = getWaitDuration(wait);
      setTiming(operation, i64Type, start, duration);
      lineReady[wait.getUpdatedLine()] = start + duration;
      makespan = std::max(makespan, start + duration);
    } else if (auto sync = mlir::dyn_cast<pulse::SyncOp>(operation)) {
      int64_t syncTime = 0;
      for (auto line : sync.getLines())
        syncTime = std::max(syncTime, lineReady.lookup(line));
      for (auto result : sync.getResults())
        lineReady[result] = syncTime;
      makespan = std::max(makespan, syncTime);
    }
  });
  return makespan;
}

static void scheduleAlap(mlir::func::FuncOp function) {
  const int64_t makespan = scheduleAsap(function);
  auto i64Type = mlir::IntegerType::get(function.getContext(), 64);
  llvm::DenseMap<mlir::Value, int64_t> lineLatest;
  llvm::SmallVector<mlir::Operation *> operations;
  function.walk([&](mlir::Operation *operation) {
    if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                  pulse::SyncOp>(operation))
      operations.push_back(operation);
  });

  auto latestFor = [&](mlir::Value line) {
    auto iterator = lineLatest.find(line);
    return iterator == lineLatest.end() ? makespan : iterator->second;
  };

  for (mlir::Operation *operation : llvm::reverse(operations)) {
    if (auto drive = mlir::dyn_cast<pulse::DriveOp>(operation)) {
      int64_t duration = getWaveformDuration(drive.getPulse());
      int64_t start = latestFor(drive.getUpdatedLine()) - duration;
      setTiming(operation, i64Type, start, duration);
      lineLatest[drive.getLine()] = start;
    } else if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(operation)) {
      int64_t duration = getWaveformDuration(readout.getPulse());
      int64_t start = latestFor(readout.getUpdatedLine()) - duration;
      setTiming(operation, i64Type, start, duration);
      lineLatest[readout.getLine()] = start;
    } else if (auto wait = mlir::dyn_cast<pulse::WaitOp>(operation)) {
      int64_t duration = getWaitDuration(wait);
      int64_t start = latestFor(wait.getUpdatedLine()) - duration;
      setTiming(operation, i64Type, start, duration);
      lineLatest[wait.getLine()] = start;
    } else if (auto sync = mlir::dyn_cast<pulse::SyncOp>(operation)) {
      int64_t syncTime = makespan;
      for (auto result : sync.getResults())
        syncTime = std::min(syncTime, latestFor(result));
      for (auto line : sync.getLines())
        lineLatest[line] = syncTime;
    }
  }
}

static size_t earliestLane(llvm::ArrayRef<int64_t> boundaries) {
  return static_cast<size_t>(
      std::distance(boundaries.begin(),
                    std::min_element(boundaries.begin(), boundaries.end())));
}

static size_t latestLane(llvm::ArrayRef<int64_t> boundaries) {
  return static_cast<size_t>(
      std::distance(boundaries.begin(),
                    std::max_element(boundaries.begin(), boundaries.end())));
}

/// Dependency- and interval-correct list scheduling with bounded drive and
/// readout resources. ``isAlap`` schedules the same dependency graph backward
/// within the makespan of the corresponding forward resource schedule.
static void scheduleRcp(mlir::func::FuncOp function, int64_t maxDrives,
                        int64_t maxReadouts, int64_t readoutLatency,
                        int64_t switchPenalty, bool isAlap) {
  if (maxDrives <= 0 || maxReadouts <= 0) {
    function.emitError("resource limits must be positive");
    return;
  }

  auto scheduleForward = [&]() {
    llvm::DenseMap<mlir::Value, int64_t> lineReady;
    llvm::SmallVector<int64_t> driveLanes(static_cast<size_t>(maxDrives), 0);
    llvm::SmallVector<int64_t> readoutLanes(static_cast<size_t>(maxReadouts),
                                            0);
    auto i64Type = mlir::IntegerType::get(function.getContext(), 64);
    int64_t makespan = 0;

    function.walk([&](mlir::Operation *operation) {
      auto place = [&](mlir::Value input, mlir::Value output, int64_t duration,
                       llvm::SmallVectorImpl<int64_t> &lanes, int64_t latency) {
        size_t lane = earliestLane(lanes);
        int64_t start = std::max(lineReady.lookup(input), lanes[lane]);
        setTiming(operation, i64Type, start, duration);
        lanes[lane] = start + duration + switchPenalty;
        lineReady[output] = start + duration + latency;
        makespan = std::max(makespan, lineReady[output]);
      };

      if (auto drive = mlir::dyn_cast<pulse::DriveOp>(operation)) {
        place(drive.getLine(), drive.getUpdatedLine(),
              getWaveformDuration(drive.getPulse()), driveLanes, 0);
      } else if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(operation)) {
        place(readout.getLine(), readout.getUpdatedLine(),
              getWaveformDuration(readout.getPulse()), readoutLanes,
              readoutLatency);
      } else if (auto wait = mlir::dyn_cast<pulse::WaitOp>(operation)) {
        int64_t start = lineReady.lookup(wait.getLine());
        int64_t duration = getWaitDuration(wait);
        setTiming(operation, i64Type, start, duration);
        lineReady[wait.getUpdatedLine()] = start + duration;
        makespan = std::max(makespan, start + duration);
      } else if (auto sync = mlir::dyn_cast<pulse::SyncOp>(operation)) {
        int64_t syncTime = 0;
        for (auto line : sync.getLines())
          syncTime = std::max(syncTime, lineReady.lookup(line));
        for (auto result : sync.getResults())
          lineReady[result] = syncTime;
        makespan = std::max(makespan, syncTime);
      }
    });
    return makespan;
  };

  const int64_t makespan = scheduleForward();
  if (!isAlap)
    return;

  auto i64Type = mlir::IntegerType::get(function.getContext(), 64);
  llvm::DenseMap<mlir::Value, int64_t> lineLatest;
  llvm::SmallVector<int64_t> driveLanes(static_cast<size_t>(maxDrives),
                                        makespan);
  llvm::SmallVector<int64_t> readoutLanes(static_cast<size_t>(maxReadouts),
                                          makespan);
  llvm::SmallVector<mlir::Operation *> operations;
  function.walk([&](mlir::Operation *operation) {
    if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                  pulse::SyncOp>(operation))
      operations.push_back(operation);
  });
  auto latestFor = [&](mlir::Value line) {
    auto iterator = lineLatest.find(line);
    return iterator == lineLatest.end() ? makespan : iterator->second;
  };

  auto place = [&](mlir::Operation *operation, mlir::Value input,
                   mlir::Value output, int64_t duration,
                   llvm::SmallVectorImpl<int64_t> &lanes, int64_t latency) {
    size_t lane = latestLane(lanes);
    int64_t end = std::min(latestFor(output) - latency, lanes[lane]);
    int64_t start = end - duration;
    setTiming(operation, i64Type, start, duration);
    lanes[lane] = start - switchPenalty;
    lineLatest[input] = start;
  };

  for (mlir::Operation *operation : llvm::reverse(operations)) {
    if (auto drive = mlir::dyn_cast<pulse::DriveOp>(operation)) {
      place(operation, drive.getLine(), drive.getUpdatedLine(),
            getWaveformDuration(drive.getPulse()), driveLanes, 0);
    } else if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(operation)) {
      place(operation, readout.getLine(), readout.getUpdatedLine(),
            getWaveformDuration(readout.getPulse()), readoutLanes,
            readoutLatency);
    } else if (auto wait = mlir::dyn_cast<pulse::WaitOp>(operation)) {
      int64_t duration = getWaitDuration(wait);
      int64_t start = latestFor(wait.getUpdatedLine()) - duration;
      setTiming(operation, i64Type, start, duration);
      lineLatest[wait.getLine()] = start;
    } else if (auto sync = mlir::dyn_cast<pulse::SyncOp>(operation)) {
      int64_t syncTime = makespan;
      for (auto result : sync.getResults())
        syncTime = std::min(syncTime, latestFor(result));
      for (auto line : sync.getLines())
        lineLatest[line] = syncTime;
    }
  }
}

template <bool IsAlap>
struct PulseSchedulePass
    : public mlir::PassWrapper<PulseSchedulePass<IsAlap>,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseSchedulePass<IsAlap>)

  llvm::StringRef getArgument() const override {
    return IsAlap ? "pulse-schedule-alap" : "pulse-schedule-asap";
  }

  llvm::StringRef getDescription() const override {
    return IsAlap ? "Assign a dependency-correct ALAP pulse schedule"
                  : "Assign a dependency-correct ASAP pulse schedule";
  }

  void runOnOperation() override {
    if constexpr (IsAlap)
      scheduleAlap(this->getOperation());
    else
      scheduleAsap(this->getOperation());
  }
};

struct PulseScheduleRcpPass
    : public mlir::PassWrapper<PulseScheduleRcpPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseScheduleRcpPass)

  PulseScheduleRcpPass(int64_t maxDrives, int64_t maxReadouts,
                       int64_t readoutLatency, int64_t switchPenalty,
                       bool isAlap)
      : maxDrives(maxDrives), maxReadouts(maxReadouts),
        readoutLatency(readoutLatency), switchPenalty(switchPenalty),
        isAlap(isAlap) {}

  llvm::StringRef getArgument() const override { return "pulse-schedule-rcp"; }
  llvm::StringRef getDescription() const override {
    return "Assign a resource-constrained pulse schedule";
  }
  void runOnOperation() override {
    scheduleRcp(getOperation(), maxDrives, maxReadouts, readoutLatency,
                switchPenalty, isAlap);
  }

  int64_t maxDrives;
  int64_t maxReadouts;
  int64_t readoutLatency;
  int64_t switchPenalty;
  bool isAlap;
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseScheduleAsapPass() {
  return std::make_unique<PulseSchedulePass<false>>();
}

std::unique_ptr<mlir::Pass> createPulseScheduleAlapPass() {
  return std::make_unique<PulseSchedulePass<true>>();
}

std::unique_ptr<mlir::Pass> createPulseScheduleRcpPass(int64_t maxDrives,
                                                       int64_t maxReadouts,
                                                       int64_t readoutLatency,
                                                       int64_t switchPenalty,
                                                       bool isAlap) {
  return std::make_unique<PulseScheduleRcpPass>(
      maxDrives, maxReadouts, readoutLatency, switchPenalty, isAlap);
}
} // namespace pulse
