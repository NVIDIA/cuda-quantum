/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstddef>
#include <optional>

namespace cudaq::opt {
namespace details {

inline bool isQuantumOperation(mlir::Operation *op) {
  return mlir::isa<quake::MeasurementInterface, quake::OperatorInterface,
                   quake::ResetOp, quake::ApplyOp>(op);
}

inline bool isBasisMeasurement(mlir::Operation *op) {
  return mlir::isa<quake::MzOp, quake::MxOp, quake::MyOp>(op);
}

inline bool containsQuantumOperation(mlir::Operation *op) {
  if (isQuantumOperation(op))
    return true;

  bool found = false;
  op->walk([&](mlir::Operation *nested) {
    if (nested == op)
      return mlir::WalkResult::advance();
    if (isQuantumOperation(nested)) {
      found = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

inline mlir::Value sampleAllocationValue(mlir::Operation *op) {
  if (!op || op->getNumResults() == 0)
    return {};
  if (auto alloca = mlir::dyn_cast<quake::AllocaOp>(op)) {
    if (alloca->hasOneUse()) {
      auto *user = *alloca->getUsers().begin();
      if (mlir::isa<quake::InitializeStateOp>(user))
        return user->getResult(0);
    }
    return alloca->getResult(0);
  }
  if (auto init = mlir::dyn_cast<quake::InitializeStateOp>(op))
    return init->getResult(0);
  return {};
}

inline std::optional<std::size_t>
extractIndexFromAllocation(mlir::Value target, mlir::Value allocation) {
  auto extract = target.getDefiningOp<quake::ExtractRefOp>();
  if (!extract || extract.getVeq() != allocation || !extract.hasConstantIndex())
    return std::nullopt;
  return extract.getConstantIndex();
}

inline bool isResetOnlyQuantumOperation(mlir::Operation *op) {
  if (mlir::isa<quake::ResetOp>(op))
    return true;

  bool foundQuantumOp = false;
  bool onlyReset = true;
  op->walk([&](mlir::Operation *nested) {
    if (nested == op || !isQuantumOperation(nested))
      return mlir::WalkResult::advance();

    foundQuantumOp = true;
    if (!mlir::isa<quake::ResetOp>(nested)) {
      onlyReset = false;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return foundQuantumOp && onlyReset;
}

inline bool hasOnlyAllowedQuantumOpsBefore(
    mlir::func::FuncOp func, mlir::Operation *firstOp, mlir::Operation *lastOp,
    const llvm::SmallPtrSetImpl<mlir::Operation *> &allowedQuantumOps) {
  if (firstOp == lastOp)
    return true;

  auto *body = &func.getBody().front();
  auto it = firstOp->getIterator();
  for (++it; it != body->end() && &*it != lastOp; ++it) {
    if (containsQuantumOperation(&*it) && !allowedQuantumOps.contains(&*it))
      return false;
  }
  return true;
}

inline bool hasOnlyResetCleanupAfter(mlir::func::FuncOp func,
                                     mlir::Operation *lastOp) {
  auto *body = &func.getBody().front();
  auto it = lastOp->getIterator();
  for (++it; it != body->end(); ++it) {
    if (containsQuantumOperation(&*it) && !isResetOnlyQuantumOperation(&*it))
      return false;
  }
  return true;
}

struct MeasurementTarget {
  mlir::Value allocation;
  std::size_t allocationOrdinal = 0;
  std::optional<std::size_t> index;
  bool wholeAllocation = false;
};

struct SeenMeasurementTarget {
  mlir::Value allocation;
  std::optional<std::size_t> index;
  bool wholeAllocation = false;
};

inline std::optional<MeasurementTarget>
classifyMeasurementTarget(mlir::Value target,
                          llvm::ArrayRef<mlir::Value> allocations) {
  for (std::size_t i = 0; i < allocations.size(); ++i) {
    auto allocation = allocations[i];
    if (target == allocation) {
      if (mlir::isa<quake::VeqType>(allocation.getType()))
        return MeasurementTarget{allocation, i, std::nullopt,
                                 /*wholeAllocation=*/true};
      return MeasurementTarget{allocation, i, std::size_t{0},
                               /*wholeAllocation=*/false};
    }

    if (auto index = extractIndexFromAllocation(target, allocation))
      return MeasurementTarget{allocation, i, index,
                               /*wholeAllocation=*/false};
  }
  return std::nullopt;
}

inline bool overlapsSeenTarget(llvm::ArrayRef<SeenMeasurementTarget> seen,
                               const MeasurementTarget &target) {
  for (const auto &previous : seen) {
    if (previous.allocation != target.allocation)
      continue;
    if (previous.wholeAllocation || target.wholeAllocation)
      return true;
    if (previous.index == target.index)
      return true;
  }
  return false;
}

inline bool followsAllocationOrder(const MeasurementTarget &target,
                                   const MeasurementTarget *previous) {
  if (!previous)
    return true;
  if (target.allocationOrdinal < previous->allocationOrdinal)
    return false;
  if (target.allocationOrdinal > previous->allocationOrdinal)
    return true;

  // Once a whole allocation is measured, another target from that allocation
  // would duplicate at least one returned bit. Individual targets in the same
  // allocation must be strictly increasing.
  if (target.wholeAllocation || previous->wholeAllocation)
    return false;
  if (!target.index || !previous->index)
    return false;
  return *target.index > *previous->index;
}

} // namespace details

inline bool requiresExplicitMeasurements(mlir::func::FuncOp func) {
  // This analysis drives sample() auto mode. It is intentionally conservative.
  // It returns false for kernels with no user measurements, or with terminal
  // basis measurements whose measured qubits form a unique stream in allocation
  // order. In those cases, the legacy/non-explicit sample path returns the same
  // global bit order, and named registers can be projected from that global
  // measurement data. Everything else returns true so sample() preserves
  // user-visible measurement order/registers with explicit measurement
  // semantics.
  if (!func || func.empty())
    return true;

  llvm::SmallVector<quake::MeasurementInterface> measurements;
  llvm::SmallVector<mlir::Value> allocations;

  // Gather all measurement operations and the local quantum allocations they
  // might measure. Extra allocations are allowed later: they may be scratch
  // qubits/vecs that do not contribute bits to the sample result.
  func.walk([&](mlir::Operation *op) {
    if (auto meas = mlir::dyn_cast<quake::MeasurementInterface>(op))
      measurements.push_back(meas);
    if (auto alloca = mlir::dyn_cast<quake::AllocaOp>(op)) {
      auto allocation = details::sampleAllocationValue(alloca);
      if (allocation)
        allocations.push_back(allocation);
    }
  });

  if (measurements.empty())
    return false;

  // If measurements exist but no local allocation is visible, the measured
  // values may be function arguments, spans, or another shape this analysis
  // cannot prove is ordered by local allocation. Require explicit semantics.
  if (allocations.empty())
    return true;

  auto *body = &func.getBody().front();
  auto *firstMeasurementOp = measurements.front().getOperation();

  auto isTopLevelMeasurement = [&](mlir::Operation *op) {
    return op->getParentOp() == func.getOperation() && op->getBlock() == body;
  };

  if (!isTopLevelMeasurement(firstMeasurementOp))
    return true;

  llvm::SmallPtrSet<mlir::Operation *, 16> allowedQuantumOps;
  llvm::SmallVector<details::SeenMeasurementTarget> seenTargets;
  std::optional<details::MeasurementTarget> previousTarget;
  mlir::Operation *lastMeasurementOp = nullptr;
  bool sawTarget = false;

  for (auto measurement : measurements) {
    auto *measurementOp = measurement.getOperation();
    if (!details::isBasisMeasurement(measurementOp) ||
        !isTopLevelMeasurement(measurementOp))
      return true;

    auto targets = measurement.getTargets();
    if (targets.empty())
      return true;

    for (auto targetValue : targets) {
      auto target =
          details::classifyMeasurementTarget(targetValue, allocations);
      if (!target)
        return true;
      if (details::overlapsSeenTarget(seenTargets, *target))
        return true;
      if (!details::followsAllocationOrder(
              *target, previousTarget ? &*previousTarget : nullptr))
        return true;

      seenTargets.push_back(details::SeenMeasurementTarget{
          target->allocation, target->index, target->wholeAllocation});
      previousTarget = *target;
      sawTarget = true;
    }

    allowedQuantumOps.insert(measurementOp);
    lastMeasurementOp = measurementOp;
  }

  if (!sawTarget || !lastMeasurementOp)
    return true;

  // Allocation-order measurements are only fast-equivalent when they are
  // terminal with respect to quantum effects. Reset cleanup after the final
  // measurement is allowed because it cannot change the returned bits, but any
  // other quantum operation between the first and last measurement, or after
  // the last measurement, means the measurement occurrence itself matters.
  if (details::hasOnlyAllowedQuantumOpsBefore(
          func, firstMeasurementOp, lastMeasurementOp, allowedQuantumOps) &&
      details::hasOnlyResetCleanupAfter(func, lastMeasurementOp))
    return false;

  return true;
}

} // namespace cudaq::opt
