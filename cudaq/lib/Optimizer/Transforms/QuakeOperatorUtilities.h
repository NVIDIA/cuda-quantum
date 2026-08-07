/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include <cassert>
#include <optional>

namespace cudaq::opt {

/// A statically selectable scalar qubit represented by a top-level Quake
/// target. A vector target records the element that must be extracted; a
/// scalar reference or wire has no element index.
struct StaticQubitTarget {
  mlir::Value source;
  std::size_t sourceIndex;
  std::optional<std::size_t> elementIndex;
};

/// Return whether \p target is a single quantum value rather than an
/// aggregate vector. This is deliberately not phase-specific: it is useful to
/// any transform that must select one qubit from a list of Quake targets.
inline bool isScalarQubitTarget(mlir::Value target) {
  return mlir::isa<cudaq::quake::RefType, cudaq::quake::WireType>(
      target.getType());
}

/// Plan a statically selectable scalar target without creating IR.
inline std::optional<StaticQubitTarget>
planStaticQubitTarget(mlir::Value target, std::size_t sourceIndex) {
  if (isScalarQubitTarget(target))
    return StaticQubitTarget{target, sourceIndex, std::nullopt};
  if (auto size = cudaq::quake::getVeqSize(target); size && *size != 0)
    return StaticQubitTarget{target, sourceIndex, *size - 1};
  return std::nullopt;
}

/// Plan the last scalar target accepted by \p predicate without creating IR.
template <typename Predicate>
inline std::optional<StaticQubitTarget>
findLastStaticQubitTarget(mlir::ValueRange targets, Predicate predicate) {
  for (std::size_t i = targets.size(); i != 0; --i) {
    auto finalTarget = planStaticQubitTarget(targets[i - 1], i - 1);
    if (!finalTarget)
      continue;
    if (!finalTarget->elementIndex) {
      if (predicate(*finalTarget))
        return finalTarget;
      continue;
    }

    for (std::size_t element = *finalTarget->elementIndex + 1; element != 0;
         --element) {
      StaticQubitTarget candidate{finalTarget->source, finalTarget->sourceIndex,
                                  element - 1};
      if (predicate(candidate))
        return candidate;
    }
  }
  return std::nullopt;
}

/// Plan a deterministic final scalar target without creating IR.
///
/// Scan source targets from right to left. A scalar reference or wire is
/// already selectable; a nonempty vector with a statically known size is
/// represented by its final element. Returning a plan rather than immediately
/// creating an ExtractRefOp lets callers finish validation before mutating the
/// IR.
inline std::optional<StaticQubitTarget>
findLastStaticQubitTarget(mlir::ValueRange targets) {
  return findLastStaticQubitTarget(
      targets, [](const StaticQubitTarget &) { return true; });
}

/// Materialize a target selected by findLastStaticQubitTarget.
///
/// Callers must perform every failure-prone validation before invoking this
/// helper, because the vector case creates an ExtractRefOp.
inline mlir::Value
materializeStaticQubitTarget(mlir::OpBuilder &builder, mlir::Location location,
                             const StaticQubitTarget &target) {
  if (!target.elementIndex)
    return target.source;
  return cudaq::quake::ExtractRefOp::create(builder, location, target.source,
                                            *target.elementIndex)
      .getResult();
}

template <typename Op>
inline llvm::SmallVector<bool> getControlPolarities(Op op) {
  llvm::SmallVector<bool> polarities(op.getControls().size(), false);
  if (auto negated = op.getNegatedQubitControls())
    for (auto [index, value] : llvm::enumerate(*negated))
      polarities[index] = value;
  return polarities;
}

/// Return the wire result types for a Quake operator with the given controls
/// and targets. Quake orders wire results by controls first, then targets.
inline llvm::SmallVector<mlir::Type>
getWireResultTypes(mlir::OpBuilder &builder, mlir::ValueRange controls,
                   mlir::ValueRange targets) {
  auto wireType = cudaq::quake::WireType::get(builder.getContext());
  llvm::SmallVector<mlir::Type> resultTypes;
  for (mlir::Value control : controls)
    if (mlir::isa<cudaq::quake::WireType>(control.getType()))
      resultTypes.push_back(wireType);
  for (mlir::Value target : targets)
    if (mlir::isa<cudaq::quake::WireType>(target.getType()))
      resultTypes.push_back(wireType);
  return resultTypes;
}

/// Update controls and targets to the corresponding wire results of a newly
/// created Quake operator.
template <typename Op>
inline void threadWireResults(Op op,
                              llvm::MutableArrayRef<mlir::Value> controls,
                              llvm::MutableArrayRef<mlir::Value> targets) {
  unsigned result = 0;
  for (mlir::Value &control : controls)
    if (mlir::isa<cudaq::quake::WireType>(control.getType()))
      control = op.getWires()[result++];
  for (mlir::Value &target : targets)
    if (mlir::isa<cudaq::quake::WireType>(target.getType()))
      target = op.getWires()[result++];
  assert(result == op.getWires().size() &&
         "gate result count does not match its wire operands");
}

/// Create a Quake gate and update its controls and targets to the latest wire
/// results. Reference operands are returned unchanged.
template <typename Op>
inline Op createAndThreadGate(mlir::OpBuilder &builder, mlir::Location location,
                              mlir::UnitAttr isAdj, mlir::ValueRange parameters,
                              llvm::MutableArrayRef<mlir::Value> controls,
                              llvm::MutableArrayRef<mlir::Value> targets,
                              mlir::DenseBoolArrayAttr negatedControls = {}) {
  auto resultTypes = getWireResultTypes(builder, controls, targets);
  auto op = Op::create(builder, location, resultTypes, isAdj, parameters,
                       controls, targets, negatedControls);
  threadWireResults(op, controls, targets);
  return op;
}

/// Collect threaded values in Quake's wire-result order.
inline llvm::SmallVector<mlir::Value> getWireValues(mlir::ValueRange controls,
                                                    mlir::ValueRange targets) {
  llvm::SmallVector<mlir::Value> values;
  for (mlir::Value control : controls)
    if (mlir::isa<cudaq::quake::WireType>(control.getType()))
      values.push_back(control);
  for (mlir::Value target : targets)
    if (mlir::isa<cudaq::quake::WireType>(target.getType()))
      values.push_back(target);
  return values;
}

} // namespace cudaq::opt
